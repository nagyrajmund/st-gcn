import sys

sys.path.append('./')

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from data.datasets import KTHDataset, SplitDataset
from data.util import loopy_pad_collate_fn
from data.augmentation import augment_data
from data.adjacency import get_normalized_adjacency_matrices
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from network import st_graphconv
from argparse import ArgumentParser
from network.st_graphconv import SpatialTemporalConv
from scripts.report.generate_figures import plot_conf_matrix

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=100,
    verbose=False,
    mode='min'
)


class L_STGCN(LightningModule):
    """
    Spatiotemporal graph-convolutional network based on ST-GCN.
    """

    def __init__(self, hparams, dataset_filters=None):
        """
        Parameters:
            hparams:  command-line arguments, see add_model_specific_args() for details
            dataset_filters:  #TODO
        """
        super().__init__()
        self.hparams = hparams
        self.dataset_filters = dataset_filters

        temporal_padding = (hparams.gamma - 1) // 2
        A = get_normalized_adjacency_matrices(hparams.partitioning, hparams.d, distance_file=hparams.distance_file)
        self.K = A.shape[0]
        self.V = A.shape[1]
        self.nr_classes = hparams.nr_classes
        self.n_layers = 10

        # TODO: check if masks are being trained
        if hparams.use_edge_importance:
            # initialise Masks for each stgcn layer as trainable parameter in network
            # TODO: this is a bit spaghetti, add a comment or create a function for this
            jitters = [ 2 * (torch.randn_like(A) - 0.5) * hparams.max_mask_jitter for i in range(self.n_layers)]
            self.Masks = nn.ParameterList([nn.Parameter( jitters[i] + torch.ones(A.shape)) for i in range(self.n_layers)])
        else:
            self.Masks = [torch.ones(A.shape) for i in range(self.n_layers)] # not trainable

        residual = self.hparams.residual
        dropout_rate = self.hparams.dropout_rate

        # Build the network
        self.conv = nn.Sequential(
            SpatialTemporalConv(hparams.C_in, 64, A * self.Masks[0], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(64, 64, A * self.Masks[1], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(64, 64, A * self.Masks[2], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(64, 64, A * self.Masks[3], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(64, 128, A * self.Masks[4], hparams.gamma, 2, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(128, 128, A * self.Masks[5], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(128, 128, A * self.Masks[6], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(128, 256, A * self.Masks[7], hparams.gamma, 2, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(256, 256, A * self.Masks[8], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual),
            SpatialTemporalConv(256, 256, A * self.Masks[9], hparams.gamma, 1, temporal_padding,
                                dropout_rate=dropout_rate, residual=residual)
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
        x = x.permute(0, 3, 1, 2)  # (N, C_in, T, V)
        x = self.conv(x)  # (N, C_out, T, V)
        N, _, T, _ = x.shape
        # Global pooling. Can't be added to Sequential as the kernel size depends on x.
        x = F.avg_pool2d(x, (x.shape[2], self.V))  # (N, C_out, 1, 1)
        x = x.view(N, x.shape[1])  # (N, C_out)
        x = self.fc_layer(x)  # (N, nr_classes)

        # x = self.softmax(x) # (N, 1)
        # don't need softmax in training if we use cross entropy as cross entropy does softmax on pred implicitly
        # see https://discuss.pytorch.org/t/making-prediction-with-argmax/49526/2
        return x

    def compute_accuracy(self, pred, gt):
        ''' pred is model output from self.forward (N, nr_classes)
             gt are categorical labels
        '''
        pred = self.softmax(pred)
        pred = torch.argmax(pred, axis=1)
        accuracy = (pred == gt).sum().item() / len(pred)
        return accuracy

    def compute_conf_mat(self, dataloader):
        confusion_matrix = torch.zeros(self.nr_classes, self.nr_classes, dtype=int)
        device = self.hparams.device

        for i, (x, y) in enumerate(dataloader):
            pred = self.forward(x)
            pred = torch.argmax(pred, axis=1)

            for i, j in zip(y, pred):
                confusion_matrix[i, j] += 1
        return confusion_matrix

    def prepare_data(self):
        if hparams.augment_data:
            print('Augmenting data')
            transforms = augment_data
        else:
            transforms = None

        # TODO (rajmund): we shouldn't duplicate the datasets
        # proposal: check if network is in training mode, if it isn't, don't augment. should be an easy fix!
        splitter = SplitDataset(self.hparams.metadata_file)

        ''' decide how to split the data for train, val, test set '''
        if self.hparams.data_split == 0:  # cross-subject training, default
            train_ind, val_ind, test_ind = splitter.split_by_subject()
        elif self.hparams.data_split == 1:  # cross-scenario training
            train_ind, val_ind, test_ind = splitter.split_by_scenario(self.hparams.train_scenarios,
                                                                      self.hparams.val_scenarios)
        else:  # stratified split
            train_ind, val_ind, test_ind = splitter.split()

        self.train_dataset = KTHDataset(metadata_csv_path=self.hparams.metadata_file,
                                        numpy_data_folder=self.hparams.dataset_dir,
                                        filter=train_ind,
                                        transforms=transforms,
                                        use_confidence_scores=False)  # TODO @rajmund:  confidence scores?

        self.val_dataset = KTHDataset(metadata_csv_path=self.hparams.metadata_file,
                                      numpy_data_folder=self.hparams.dataset_dir,
                                      filter=val_ind,
                                      transforms=None,  # No augmentation during validation!
                                      use_confidence_scores=False)

        self.test_dataset = KTHDataset(metadata_csv_path=self.hparams.metadata_file,
                                       numpy_data_folder=self.hparams.dataset_dir,
                                       filter=test_ind,
                                       transforms=None,  # No augmentation during testing!
                                       use_confidence_scores=False)

        self.train_sampler = RandomSampler(self.train_dataset)
        # TODO: it's best practice to not shuffle the dataset for validation and testing
        # we'll use subsetsampler when stop duplicating the datasets
        """
        self.val_sampler = RandomSampler(self.val_dataset)
        self.test_sampler = RandomSampler(self.test_dataset)
        """

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          sampler=self.train_sampler, collate_fn=loopy_pad_collate_fn,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=loopy_pad_collate_fn,
                          num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          collate_fn=loopy_pad_collate_fn,
                          num_workers=self.hparams.num_workers, shuffle=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        acc = self.compute_accuracy(output, y)
        print('loss: ', loss)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = sum([x['acc'] for x in outputs]) / len(outputs)
        tensorboard_logs = {'train_loss': avg_loss, 'train_acc': avg_accuracy}
        return {'avg_train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        val_acc = self.compute_accuracy(output, y)
        return {'loss': loss, 'acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = sum([x['acc'] for x in outputs]) / len(outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_accuracy}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def on_train_end(self):
        # print('Saving Confusion matrix')
        print('Unable to save confusion matrix')
        # confusion_matrix = self.compute_conf_mat(self.val_dataloader())
        # fig = plot_conf_matrix(confusion_matrix)
        # self.logger.experiment.add_figure( \
        #     "Confusion matrix for validation set", fig, global_step=None, close=True, walltime=None)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        test_acc = self.compute_accuracy(output, y)
        return {'loss': loss, 'acc': test_acc}

    def test_epoch_end(self, outputs):
        # averages across epoch
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = sum([x['acc'] for x in outputs]) / len(outputs)
        tensorboard_logs = {'avg_test_loss': avg_loss, 'avg_test_acc': avg_accuracy}

        # print('Saving Confusion matrix')
        # confusion_matrix = self.compute_conf_mat(self.test_dataloader())
        # fig = plot_conf_matrix(confusion_matrix)
        # self.logger.experiment.add_figure( \
        #     "Confusion matrix for test set", fig, global_step=None, close=True, walltime=None)

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--augment_data', type=bool, default=False,
                            help='if passed in, performs random augmentation during training (as in the paper)')
        parser.add_argument('--d', type=int, default=1, help='max distance in spatial neighbourhood')
        parser.add_argument('--gamma', type=int, default=9, help='temporal kernel size')
        parser.add_argument('--dropout_rate', type=float, default=0, help='dropout thresh')
        # edge importance args
        parser.add_argument('--use_edge_importance', type=bool, default=False,
                            help='if passed in, uses learnable edge importance masks')
        parser.add_argument('--max_mask_jitter', type=float, default=0.001,
                            help="maximal amount of random perturbation added to the initial edge importance masks")

        parser.add_argument('--partitioning', type=int, default=0,
                            help='partitioning strategy (0 - unilabeling, 1 - distance labeling, 2 - spatial partitioning, 3 - symmetrical-distance labeling)')
        parser.add_argument('--residual', type=bool, default=False, help='if passed in, uses residual block')
        # data split args
        parser.add_argument('--data_split', type=int, default=0,
                            help='way to split the data into train/val/test sets (0 - cross-subject, 1 - cross-scenario, 2 - ordinary stratified')
        parser.add_argument('--train_scenarios', type=list, default=["d1", "d2"],
                            help='scenarios to put into the training set (list of any from d1,d2,d3,d4)')
        parser.add_argument('--val_scenarios', type=list, default=["d3"],
                            help='scenarios to put into the training set (list of any from d1,d2,d3,d4)')
        # early stopping takes min epochs into consideration
        # parser.add_argument('--min_epochs', type=int, default=500, help='min epochs to run the network for.')
        # parser.add_argument('--early_stop', type=bool, default=False, help='use early stopping during training')
        # parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use (adam, sgd)')
        # TODO rajmund: confidence score, optimizer type missing
        return parser


def build_argument_parser():
    parser = ArgumentParser()
    # Add program level args
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--nr_classes', type=int, default=6)
    parser.add_argument('--metadata_file', type=str, default='../datasets/KTH_Action_Dataset/metadata.csv',
                        help='path to the .csv file that contains the metadata')
    parser.add_argument('--dataset_dir', type=str, default='../datasets/KTH_Action_Dataset', help='path to the dataset')
    parser.add_argument('--C_in', type=int, default=2, help='number of channels in the data')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--distance_file', type=str, default='')
    parser.add_argument('--use_early_stopping', type=bool, default=False)

    parser = L_STGCN.add_model_specific_args(parser)  # Add model-specific args
    parser = Trainer.add_argparse_args(parser)  # Add ALL training-specific args

    # In case of spatial conf. partitioning, pre-calculate distances and store them in a file

    return parser


if __name__ == "__main__":
    # Parse command-line args
    hparams = build_argument_parser().parse_args()

    model = L_STGCN(hparams)
    # TODO: check Trainer args: gradient clipping, amp_level for 16-bit precision etc
    # TODO: tidy up and add min epochs, patience etc to args
    if hparams.use_early_stopping:
        print('Using early stopping')
        trainer = Trainer.from_argparse_args(hparams, early_stop_callback=early_stop_callback, min_epochs=500)
    else:
        print('Not using early stopping')
        trainer = Trainer.from_argparse_args(hparams)

    trainer.fit(model)
    trainer.test()
    print('Saving logs')
    trainer.save_checkpoint("../models/final_model.ckpt")

    # load model back in
#     trainer = Trainer.from_argparse_args(hparams)
#     checkpoint_model = L_STGCN.load_from_checkpoint(checkpoint_path="../models/final_model.ckpt")
#     trainer.fit(checkpoint_model)
#     checkpoint_model.on_test_end()


