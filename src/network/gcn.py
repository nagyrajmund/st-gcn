import torch.nn as nn
from adjacency import Strategy
import adjacency as adj

class GCN(nn.Module):
    """
    Spatial convolutional layer.
    """

    def __init__(self, C_in, C_out, V, T,
                kernel_size = (1, 1), stride = (1, 1), padding = (0, 0), dilation = (1, 1), bias = True,
                strat = Strategy.UNI_LABELING):

        # TODO spatial conf.: pre-calculate the distances or pass training data and calculate them here?
        # Do we need to create A and K inside the constructor as opposed to just passing them?
        super(GCN, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.V = V
        self.T = T
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.A = adj.create_adjacency_matrices(strat = strat)
        self.A = np.asarray(adj.normalize(self.A))
        self.K = len(self.A)

        self.W = nn.Conv2D(in_channels = C_in,
                           out_channels = C_out * K,
                           kernel_size = kernel_size,
                           stride = stride,
                           padding = padding,
                           dilation = dilation,
                           bias = bias)

    def forward(self, f_in):
        f_in = self.W(f_in) # Dimension is (K, C_out, T, V)?
        f_out = np.tensordot(self.A, f_in)
        # TODO Sum over K
        return f_out