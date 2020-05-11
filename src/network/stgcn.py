import torch.nn as nn
from .st_graphconv import SpatialTemporalConv
import torch.nn.functional as F
from data import adjacency as adj
import torch


class STGCN(nn.Module):
    """
    Network containing all the layers needed for spatio-temporal graph convolution.
    """

    def __init__(self, C_in, gamma, nr_classes, strat = adj.Strategy.UNI_LABELING, d = 1, edge_importance=True):
        """
        Parameters:
            C_in:  number of input channels
            gamma:  kernel size for the temporal convolution
            nr_classes:  number of classes
            strat:  partitioning strategy (optional)
            d:  distance (optional)
            edge_importance: whether to use edge importance weighting (optional, default True)
        """

        super().__init__()

        self.nr_classes = nr_classes
        temporal_padding = (gamma - 1) // 2
        A = torch.Tensor(adj.get_normalized_adjacency_matrices(strat, d))
        self.K = A.shape[0]
        self.V = A.shape[1]
        self.C_in = C_in
        self.C_out = 256

        if edge_importance:
            # initialise Masks for each stgcn layer as trainable parameter in network
            self.Masks = torch.nn.ParameterList([nn.Parameter(torch.ones(A.shape)) for i in range(10)])
        else:
            self.Masks = [torch.ones(A.shape) for i in range(10)] # not trainable


        self.conv = nn.Sequential(
            SpatialTemporalConv(self.C_in, 64, A*self.Masks[0], gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A*self.Masks[1], gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A*self.Masks[2], gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A*self.Masks[3], gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 128, A*self.Masks[4], gamma, 2, temporal_padding),
            SpatialTemporalConv(128, 128, A*self.Masks[5], gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 128, A*self.Masks[6], gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 256, A*self.Masks[7], gamma, 2, temporal_padding),
            SpatialTemporalConv(256, 256, A*self.Masks[8], gamma, 1, temporal_padding),
            SpatialTemporalConv(256, self.C_out, A*self.Masks[9], gamma, 1, temporal_padding)
        ).float()

        self.fc_layer = nn.Linear(256, self.nr_classes).float()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x:  (N, T, V, C_in)

        Returns:
            the results of classification
        """
        x = x.permute(0, 3, 1, 2)  # reshape for forward algorithm to shape (N, C_in, T, V)

        # Appy convolutional layers.
        x = self.conv(x) # (N, C_out, T, V)
        N, _, T, _ = x.shape

        # Global pooling. Can't be added to Sequential as the kernel size depends on x.
        x = F.avg_pool2d(x, (T, self.V)) # (N, C_out, 1, 1)
        x = x.view(N, self.C_out) # (N, C_out)
        # Fully connected layer + SoftMax
        x = self.fc_layer(x) # (N, nr_classes)
        x = self.softmax(x) # (N, 1)
        # don't need softmax if we use cross entropy as cross entropy does softmax on pred implicitly
        # see https://discuss.pytorch.org/t/making-prediction-with-argmax/49526/2
        return x
