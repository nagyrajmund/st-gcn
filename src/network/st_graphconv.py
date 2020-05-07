import torch.nn as nn
from adjacency import Strategy
import adjacency as adj

class SpatialTemporalConv(nn.Module):
    """
    Perform spatial and temporal convolution on a sequence of joint locations.
    """

    def __init__(self, C_in, C_out, A, gamma, spatial_args, temporal_args):
        """
        Parameters:
            C_in:  number of input channels
            C_out:  number of output channels
            A:  normalized adjacency matrix (K,V,V) where K is the spatial kernel size
            gamma:  kernel size for the temporal convolution
            spatial_args:  optional keyword arguments for the spatial Conv2D layer as a dict
            temporal_args:  optional keyword arguments for the temporal Conv2d layer as a dict
        
        NOTE: spatial_args and temporal_args should not contain the channel and kernel sizes, as they 
        are deduced automatically! For details about the possible parameters, see the official PyTorch documentation.
        """
        super().__init__()
        
        self.spatialConv = SpatialConv(C_in, C_out, A, spatial_args)
        temporal_kernel_size = (1, gamma)
        self.temporalConv = nn.Conv2d(C_out, C_out, temporal_kernel_size, **temporal_args)
    
    def forward(self, f_in):
        # TODO check if we need to reshape the dimensions
        return self.temporalConv(self.spatialConv(f_in))

class SpatialConv(nn.Module):
    """
    Spatial convolutional layer.
    """

    def __init__(self, C_in, C_out, A, spatial_args):
        """
        Constructs the layer with given parameters.
        
        Parameters:
            C_in:  number of input channels
            C_out:  number of output channels
            A:  normalized adjacency matrix (K, V, V)
            kernel_size:  kernel size of the 2D convolution (optional)
            
            spatial_args:  optional keyword arguments for the spatial Conv2D layer as a dict
        
        NOTE: spatial_args should not contain the channel and kernel sizes, as they 
        are deduced automatically! For details about the possible parameters, see the official PyűTorch documentation.
        """

        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.A = A
        self.K = self.A.shape[0]
        self.V = self.A.shape[1]
        # Filter size is 1x1 (so we that don't reduce the input size), but the
        # Output has to be K, C_out, T, V
        kernel_size = (1,1) 
        self.W = nn.Conv2d(C_in, self.K * C_out, kernel_size, **spatial_args)

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