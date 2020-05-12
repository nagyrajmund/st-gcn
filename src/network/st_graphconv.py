import torch.nn as nn
import torch

class SpatialTemporalConv(nn.Module):
    """
    Perform spatial and temporal convolution on a sequence of joint locations.
    """

    # TODO @livia Not sure that some layers are in the correct order? (Batch normalization, ReLU)
    def __init__(self, C_in, C_out, A, gamma, temporal_stride, temporal_padding, dropout_rate=0.5):
        """
        Parameters:
            C_in:  number of input channels
            C_out:  number of output channels
            A:  normalized adjacency matrix (K,V,V) where K is the spatial kernel size
            gamma:  kernel size for the temporal convolution
            temporal_stride:  stride for the temporal layer
            temporal_padding:  padding for the temporal layer
            dropout_rate:  probability of an element to be zeroed (optional)
        """

        super().__init__()

        # Batch normalization
        # TODO Is it okay to use this here? According to the paper,
        # "In our experiments, we first feed input skeletons to a batch normalization layer to normalize data."
        self.batch_n = nn.BatchNorm2d(C_in)

        # Spatial convolution
        self.spatialConv = SpatialConv(C_in, C_out, A)

        # Temporal convolution
        temporal_kernel_size = (gamma, 1)
        self.temporalConv = \
            nn.Conv2d(C_out, C_out, kernel_size=temporal_kernel_size, stride=(temporal_stride, 1),
                  padding=(temporal_padding, 0))

        # Batch normalization
        self.batch_n_2 = nn.BatchNorm2d(C_out)

        # Activation
        self.relu = nn.ReLU(inplace = True)

        # Dropout
        # "And we randomly dropout the features at 0.5 probability after each STGCN unit to avoid overfitting."
        self.dropout = nn.Dropout(dropout_rate, inplace = True)

    def forward(self, f_in):
        """
        Forward pass.

        Parameters:
            f_in:  input of size (N, C_in, T, V)

        Returns:
            output of the ST-GCN unit
        """

        f_in = self.batch_n(f_in.float())
        f_out = self.temporalConv(self.spatialConv(f_in)) # (N, C_out, T, V)
        f_out = self.batch_n_2(f_out)
        # relu and dropout are inplace operations so have to clone input and rename variables to allow variables \
        # to be accessible for backward algorithm.
        # we don't set inplace = False as this typically decreases performance
        f_act = self.relu(f_out.clone())
        f_drop = self.dropout(f_act.clone())
        return f_drop

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
        self.A = nn.Parameter(A.float()) # TODO parameter or register buffer? https://discuss.pytorch.org/t/resolved-runtimeerror-expected-device-cpu-and-dtype-float-but-got-device-cuda-0-and-dtype-float/54783/18
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
        self.A = self.A.to(f_in.device) # TODO needed?

        N, C, T, _ = f_in.shape # Number of data points, number of channels, number of time frames
        f_in = self.W(f_in) # Dimension is (N, K * C_out, T, V)
        f_in = f_in.view(N, self.K, self.C_out, T, self.V) # TODO Order preserved?
        f_out = torch.einsum('kvw,nkctw->nctv', self.A, f_in)

        return f_out

