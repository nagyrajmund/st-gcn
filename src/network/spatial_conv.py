import torch.nn as nn
from adjacency import Strategy
import adjacency as adj

class SpatialConv(nn.Module):
    """
    Spatial convolutional layer.
    """

    def __init__(self, C_in, C_out, A,
                kernel_size = (1, 1), stride = (1, 1), padding = (0, 0), dilation = (1, 1), bias = True):
        """
        Constructs the layer with given parameters.
        
        Parameters:
            C_in:  number of input channels
            C_out:  number of output channels
            A:  normalized adjacency matrix (K, V, V)
            kernel_size:  kernel size of the 2D convolution (optional)
            stride:  stride of the 2D convolution (optional)
            padding:  padding of the 2D convolution (optional)
            dilation:  dilation of the 2D convolution (optional)
            bias:  if True, use bias in the 2D convolution (optional)
        """

        super(SpatialConv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.A = A
        self.K = self.A.shape[0]
        self.V = self.A.shape[1]

        self.W = nn.Conv2D(in_channels = C_in,
                           out_channels = C_out * K,
                           kernel_size = kernel_size,
                           stride = stride,
                           padding = padding,
                           dilation = dilation,
                           bias = bias)

    def forward(self, f_in):
        """
        Forward pass.

        f_in: input data of size (N, C_in, T, V)
        """

        N, T = f_in.shape[0], f_in.shape[2] # Number of data points, number of time frames

        f_in = self.W(f_in) # Dimension is (N, K * C_out, T, V)
        f_in = f_in.view(N, self.K, self.C_out, T, self.V) # TODO Order?

        f_out = np.einsum('kvw,nkctw->nctv', self.A, f_in)
        return f_out