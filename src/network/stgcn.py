import torch.nn as nn
from st_graphconv import SpatialTemporalConv
import adjacency as adj
from adjacency import Strategy

class STGCN(nn.Module):
    """
    Network containing all the layers needed for spatio-temporal graph convolution.
    """

    def __init__(self, C_in, gamma, strat = Strategy.UNI_LABELING, d = 1):
        """
        Parameters:
            C_in:  number of input channels
            gamma:  kernel size for the temporal convolution
            strat:  partitioning strategy (optional)
            d:  distance (optional)
        """

        super().__init__()

        temporal_padding = (gamma - 1) / 2
        A = adj.get_normalized_adjacency_matrices(strat, d)

        self.model = nn.Sequential(
            SpatialTemporalConv(C_in, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 64, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(64, 128, A, gamma, 2, temporal_padding),
            SpatialTemporalConv(128, 128, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 128, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 256, A, gamma, 2, temporal_padding),
            SpatialTemporalConv(256, 256, A, gamma, 1, temporal_padding),
            SpatialTemporalConv(256, 256, A, gamma, 1, temporal_padding)
        )

    def forward(self, x):
        """
        Forward pass.
        x: (N, T, V, C_in)

        f_in: input data of size (N, C_in, T, V)
        """
        f_in = x.permute(0, 3, 1, 2)  # reshape for forward algorithm to shape (N, C, T, V)
        self.model(f_in)
