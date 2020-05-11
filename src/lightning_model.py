import sys
sys.path.append('./')

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from data.datasets import KTHDataset, SplitDataset
from data.util import loopy_pad_collate_fn
from data.augmentation import augment_data
from data.adjacency import get_normalized_adjacency_matrices
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from network import st_graphconv
from argparse import ArgumentParser
from network.st_graphconv import SpatialTemporalConv

class L_STGCN(LightningModule):
    retain_graph = True
    def __init__(self, hparams, dataset_filters=None):
        #TODO @rajmund:  update documentation about parameters once it's stable
        super().__init__() 
        self.hparams = hparams
        self.dataset_filters = dataset_filters

        temporal_padding = (hparams.gamma - 1) // 2
        A = torch.Tensor(get_normalized_adjacency_matrices(hparams.partitioning, hparams.d))
        self.K = A.shape[0]
        self.V = A.shape[1]

        if hparams.edge_importance: #TODO rename
            # initialise Masks for each stgcn layer as trainable parameter in network
            self.Masks = nn.ParameterList([nn.Parameter(torch.ones(A.shape)) for i in range(10)])
        else:
            self.Masks = [torch.ones(A.shape) for i in range(10)] # not trainable
        
        # Build the network
        self.conv = nn.Sequential(
            SpatialTemporalConv(hparams.C_in, 64, A*self.Masks[0], hparams.gamma, 1, temporal_padding),
            SpatialTemporalConv( 64,  64, A*self.Masks[1], hparams.gamma, 1, temporal_padding),
            SpatialTemporalConv( 64,  64, A*self.Masks[2], hparams.gamma, 1, temporal_padding),
            SpatialTemporalConv( 64,  64, A*self.Masks[3], hparams.gamma, 1, temporal_padding),
            SpatialTemporalConv( 64, 128, A*self.Masks[4], hparams.gamma, 2, temporal_padding),
            SpatialTemporalConv(128, 128, A*self.Masks[5], hparams.gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 128, A*self.Masks[6], hparams.gamma, 1, temporal_padding),
            SpatialTemporalConv(128, 256, A*self.Masks[7], hparams.gamma, 2, temporal_padding),
            SpatialTemporalConv(256, 256, A*self.Masks[8], hparams.gamma, 1, temporal_padding),
            SpatialTemporalConv(256, 256, A*self.Masks[9], hparams.gamma, 1, temporal_padding)
        ).float()

        self.fc_layer = nn.Linear(256, hparams.nr_classes).float()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x:  (N, T, V, C_in)

        Returns:
            the results of classification
        """
        x = x.permute(0, 3, 1, 2) # (N, C_in, T, V)  
        N, _, T, _ = x.shape

        x = self.conv(x) # (N, C_out, T, V)
        
        # Global pooling. Can't be added to Sequential as the kernel size depends on x.
        x = F.avg_pool2d(x, (T, self.V)) # (N, C_out, 1, 1)
        x = x.view(N, self.C_out) # (N, C_out)
        
        x = self.fc_layer(x) # (N, nr_classes)
        # x = self.softmax(x) # (N, 1)
        # don't need softmax if we use cross entropy as cross entropy does softmax on pred implicitly
        # see https://discuss.pytorch.org/t/making-prediction-with-argmax/49526/2
        return x

    def prepare_data(self):
        if hparams.augment_data:
            transforms = augment_data
        else:
            transforms = None
        
        self.train_dataset = KTHDataset(metadata_csv_path = self.hparams.metadata_file,
                                        numpy_data_folder = self.hparams.dataset_dir,
                                        #filter = self.dataset_filters['train'], #TODO @rajmund: where do we construct this?
                                        transforms = transforms,
                                        use_confidence_scores = False) #TODO @rajmund:  confidence scores?

        self.val_dataset   = KTHDataset(metadata_csv_path = self.hparams.metadata_file,
                                        numpy_data_folder = self.hparams.dataset_dir,
                                        #filter = self.dataset_filters['val'], 
                                        transforms = transforms,
                                        use_confidence_scores = False)

        self.test_dataset  = KTHDataset(metadata_csv_path = self.hparams.metadata_file,
                                        numpy_data_folder = self.hparams.dataset_dir,
                                        #filter = self.dataset_filters['test'], 
                                        transforms = None, # No augmentation during testing!
                                        use_confidence_scores = False) 
        
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = RandomSampler(self.val_dataset)
        self.test_sampler = RandomSampler(self.test_dataset)
    
    def train_dataloader(self):
       return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                         sampler=self.train_sampler, collate_fn=loopy_pad_collate_fn)

    # def val_dataloader(self):
    #    return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
    #                      sampler=self.val_sampler)

    # def test_dataloader(self):
    #    return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
    #                      sampler=self.test_sampler)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)

        logs = {'loss' : loss}
        return {'loss': loss, 'log': logs}

    def backward(self, use_amp, loss, optimizer):
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph = self.retain_graph)
        else:
            loss.backward(retain_graph = self.retain_graph)

        self.retain_graph = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--augment_data', type=bool, default=False)
        parser.add_argument('--d', type=int, default=1)
        parser.add_argument('--gamma', type=int, default=9)
        parser.add_argument('--edge_importance', type=bool, default=True)
        parser.add_argument('--partitioning', type=int, default=0) 
        # Partitioning: 0 for unilabeling, 1 for distance and 2 for spatial

        #TODO rajmund: confidence score, optimizer type missing
        return parser

# TODO @rajmund:  missing use_confidence_scores in parameters
# training_config =
# {
#     'device': device,
#     'dataset_dir': Path(dataset_dir),
#     'metadata_file': Path(dataset_dir) / 'metadata.csv',
#     'batch_size': 8,
#     'n_epochs': 20
# }

def build_argument_parser():
    parser = ArgumentParser()
    # Add program level args
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--nr_classes', type=int, default=6)
    parser.add_argument('--metadata_file', type=str, default='../datasets/KTH_Action_Dataset/metadata.csv')
    parser.add_argument('--dataset_dir', type=str, default='../datasets/KTH_Action_Dataset')
    parser.add_argument('--C_in', type=int, default=2)

    parser = L_STGCN.add_model_specific_args(parser) # Add model-specific args
    parser = Trainer.add_argparse_args(parser) # Add ALL training-specific args
    
    return parser

if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()
    model = L_STGCN(hparams)
    #TODO: check Trainer args: gradient clipping, amp_level for 16-bit precision etc
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model)
