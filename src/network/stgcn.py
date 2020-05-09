import torch.nn as nn
from .st_graphconv import SpatialTemporalConv
import torch.nn.functional as F
from . import adjacency as adj


class STGCN(nn.Module):
    """
    Network containing all the layers needed for spatio-temporal graph convolution.
    """

    def __init__(self, C_in, gamma, nr_classes, strat = adj.Strategy.UNI_LABELING, d = 1):
        """
        Parameters:
            C_in:  number of input channels
            gamma:  kernel size for the temporal convolution
            nr_classes:  number of classes
            strat:  partitioning strategy (optional)
            d:  distance (optional)
        """

        super().__init__()

        self.nr_classes = nr_classes
        temporal_padding = (gamma - 1) // 2
        A = adj.get_normalized_adjacency_matrices(strat, d)
        self.K = A.shape[0]
        self.V = A.shape[1]
        self.C_in = C_in
        self.C_out = 256

        self.conv = nn.Sequential(
            SpatialTemporalConv(self.C_in, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 128, A, gamma, 2, temporal_padding),
            SpatialTemporalConv(128, 128, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 128, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 256, A, gamma, 2, temporal_padding),
            SpatialTemporalConv(256, 256, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(256, self.C_out, A, gamma, 1, temporal_padding)
        ).double()

        self.fc_layer = nn.Linear(256, self.nr_classes).double()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass.
        x: (N, T, V, C_in)

        Parameters:
            f_in: input data of size (N, C_in, T, V)

        Returns:
            the results of classification
        """
        x = x.permute(0, 3, 1, 2)  # reshape for forward algorithm to shape (N, C, T, V)

        # Appy convolutional layers.
        x = self.conv(x) # (N, C_out, T, V)
        N, _, T, _ = x.shape

        # Global pooling. Can't be added to Seqential as the kernel size depends on x.
        x = F.avg_pool2d(x, (T, self.V)) # (N, C_out, 1, 1)
        x = x.view(N, self.C_out) # (N, C_out)
        # Fully connected layer + SoftMax
        x = self.fc_layer(x) # (N, nr_classes)
        x = self.softmax(x) # (N, 1)
        # TODO @amrita remove softmax, don't need if we use cross entropy as cross entropy does softmax on pred implicitly
        # see https://discuss.pytorch.org/t/making-prediction-with-argmax/49526/2
        return x
