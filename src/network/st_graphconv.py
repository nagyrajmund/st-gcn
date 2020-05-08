import torch.nn as nn
from torch import einsum, Tensor

class SpatialTemporalConv(nn.Module):
    """
    Perform spatial and temporal convolution on a sequence of joint locations.
    """

    def __init__(self, C_in, C_out, A, gamma, temporal_stride, temporal_padding):
        """
        Parameters:
            C_in:  number of input channels
            C_out:  number of output channels
            A:  normalized adjacency matrix (K,V,V) where K is the spatial kernel size
            gamma:  kernel size for the temporal convolution
            temporal_stride:  stride for the temporal layer
            temporal_padding:  padding for the temporal layer
        """

        super().__init__()

        self.spatialConv = SpatialConv(C_in, C_out, A)

        temporal_kernel_size = (gamma, 1)
        self.temporalConv = \
            nn.Conv2d(C_out, C_out, kernel_size=temporal_kernel_size, stride=(temporal_stride, 1),
                  padding=(temporal_padding, 0))

    def forward(self, f_in):
        f_out = self.temporalConv(self.spatialConv(f_in)) # (N, C_out, T, V)
        return f_out

class SpatialConv(nn.Module):
    """
    Spatial convolutional layer. Performs 1x1 convolution with the (1,1,C_in,C_out,K)-dimensional filter.
    """

    def __init__(self, C_in, C_out, A):
        """
        Constructs the layer with given parameters.

        Parameters:
            C_in:  number of input channels
            C_out:  number of output channels
            A:  normalized adjacency matrix (K, V, V)
        """

        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.A = Tensor(A).double()
        self.K = self.A.shape[0]
        self.V = self.A.shape[1]
        # Filter size is 1x1 (so we that don't reduce the input size), but the
        # Output has to be K, C_out, T, V
        kernel_size = (1, 1)

        self.W = nn.Conv2d(C_in, self.K * C_out, kernel_size)

    def forward(self, f_in):
        """
        Forward pass.

        f_in: input data of size (N, C_in, T, V)
        """
        N, C, T, _ = f_in.shape # Number of data points, number of channels, number of time frames
        f_in = self.W(f_in) # Dimension is (N, K * C_out, T, V)
        f_in = f_in.view(N, self.K, self.C_out, T, self.V) # TODO Order preserved?
        f_out = einsum('kvw,nkctw->nctv', self.A, f_in)

        return f_out

