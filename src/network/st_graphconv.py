import torch.nn as nn
import torch

class SpatialTemporalConv(nn.Module):
    """
    Perform spatial and temporal convolution on a sequence of joint locations.
    """

    def __init__(self, C_in, C_out, A, gamma, temporal_stride, temporal_padding, dropout_rate=0.5, residual=False):
        """
        Parameters:
            C_in:  number of input channels
            C_out:  number of output channels
            A:  normalized adjacency matrix (K,V,V) where K is the spatial kernel size
            gamma:  kernel size for the temporal convolution
            temporal_stride:  stride for the temporal layer
            temporal_padding:  padding for the temporal layer
            dropout_rate:  probability of an element to be zeroed (optional)
            residual: whether to apply residual block
        """

        super().__init__()

        if residual:
            if C_in == C_out and temporal_stride == 1: # shapes match
                self.apply_residual = lambda x: x
            else:
                self.apply_residual = nn.Conv2d(C_in, C_out, kernel_size=1, stride=(temporal_stride, 1))

        self.residual = residual

        # Batch normalization
        # "In our experiments, we first feed input skeletons to a batch normalization layer to normalize data."
        self.batch_n = nn.BatchNorm2d(C_in)

        # Spatial convolution
        self.spatialConv = SpatialConv(C_in, C_out, A)

        # Temporal convolution
        temporal_kernel_size = (gamma, 1)
        self.temporalConv = \
            nn.Conv2d(C_out, C_out, kernel_size=temporal_kernel_size, stride=(temporal_stride, 1),
                  padding=(temporal_padding, 0))

        # Batch normalization after temporal convolution (or before temporal convolution if residual is used)
        self.batch_n_2 = nn.BatchNorm2d(C_out)

        # Activation
        self.relu = nn.ReLU(inplace = True)

        # Dropout
        # "And we randomly dropout the features at 0.5 probability after each STGCN unit to avoid overfitting."
        if dropout_rate != 0:
            print('Using dropout')
            self.dropout = nn.Dropout(dropout_rate, inplace = True)
        else:
            print('Not using dropout')
            self.dropout = None

    def residual_block(self, f_in):
        '''
        Performs full-preactivation residual block as follows:
        BN -> activation -> Weights -> BN -> activation -> Weights
        See (e) full pre-activation in https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

        Parameters:
            f_in: input of size (N, C_in, T, V)
        Returns:
            output of residual block after full-preactivation
        '''
        res = f_in.clone()
        # first set of pre-activation
        f_in = self.batch_n(f_in.float())
        f_act = self.relu(f_in.clone())
        f_out = self.spatialConv(f_act)
        # second set of pre-activation
        f_in = self.batch_n_2(f_out.float())
        f_act = self.relu(f_in.clone())
        f_out = self.temporalConv(f_act) # (N, C_out, T, V)
        # Add residual
        f_out += self.apply_residual(res.float())
        return f_out


    def forward(self, f_in):
        """
        Forward pass.

        Parameters:
            f_in:  input of size (N, C_in, T, V)

        Returns:
            output of the ST-GCN unit
        """
        if self.residual:
            f_out = self.residual_block(f_in)
        else:
            f_in = self.batch_n(f_in.float())
            f_out = self.temporalConv(self.spatialConv(f_in)) # (N, C_out, T, V)
            f_out = self.batch_n_2(f_out)

        # relu and dropout are inplace operations so have to clone input and rename variables to allow variables \
        # to be accessible for backward algorithm.
        # we don't set inplace = False as this typically decreases performance
        f_act = self.relu(f_out.clone())
        if self.dropout is None:
            return f_act
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
        f_in = f_in.view(N, self.K, self.C_out, T, self.V)
        f_out = torch.einsum('kvw,nkctw->nctv', self.A, f_in)

        return f_out

