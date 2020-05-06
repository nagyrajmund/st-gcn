import torch.nn as nn

class STGCN(nn.Module):
    """
    Layer encapsulating both the spatial and the temporal aspect.
    """
     
    def __init__(self):
        super(STGCN, self).__init__()