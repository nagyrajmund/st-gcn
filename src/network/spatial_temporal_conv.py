import torch.nn as nn

class SpatialTemporalConv(nn.Module):
    """
    Layer encapsulating the spatial and the temporal aspect.
    """

    def __init__(self):
        super(SpatialTemporalConv, self).__init__()