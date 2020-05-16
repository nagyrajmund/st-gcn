# TODO: this should be added to every runnable script to make imports work
import sys
sys.path.append('../')

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from sklearn.model_selection import train_test_split
from data import util
from data.augmentation import augment_data
import random

class SplitDataset:
    def __init__(self, metadata_csv_path):
        self.metadata = pd.read_csv(metadata_csv_path)

    def _get_indices(self, train, val, test):
        return list(train.index.values), list(val.index.values), list(test.index.values)

    def split_by_subject(self, train=15, val=5, test=5, randomise_split=False):
        ''' returns: indices corresponding to train and test, with train/(train+test) subjects in the train set,
            test/(train+test) subjects in the test set.
            randomise_split: boolean, if true, subjects are randomly allocated to train, val, test set
        '''
        assert train + val + test == 25
        subjects = list(set(self.metadata['subject']))
        subjects.sort()
        # get subjects in train, val and test sets

        if randomise_split: # randomise split of subjects into train, val, test set
            train_subjects, val_test_subjects = train_test_split(subjects, train_size=train/(train+val+test))
            val_subjects, test_subjects = train_test_split(val_test_subjects, train_size=val/(val+test))
        else:
            train_subjects = subjects[:train]
            val_subjects = subjects[train:train + val]
            test_subjects = subjects[train + val:]

        train = self.metadata[self.metadata['subject'].isin(train_subjects)]
        val = self.metadata[self.metadata['subject'].isin(val_subjects)]
        test = self.metadata[self.metadata['subject'].isin(test_subjects)]

        # get indices of videos with these subjects in metadata_csv file
        train_indices, val_indices, test_indices = self._get_indices(train, val, test)

        assert (len(train_indices) + len(val_indices) + len(test_indices)) == len(self.metadata)
        return train_indices, val_indices, test_indices


    def split_by_scenario(self, train_scenarios, val_scenarios):
        ''' returns: indices corresponding to train and test, with as close to train/(train+test) subjects in the train set as possible,
            test/(train+test) subjects in the test set.
        '''
        train = self.metadata[self.metadata['scenario'].isin(train_scenarios)]
        val = self.metadata[self.metadata['scenario'].isin(val_scenarios)]
        test = self.metadata[~self.metadata['scenario'].isin(val_scenarios+train_scenarios)]

        train_indices, val_indices, test_indices = self._get_indices(train, val, test)

        assert (len(train_indices) + len(val_indices) + len(test_indices)) == len(self.metadata)
        return train_indices, val_indices, test_indices

    def split(self, data=None, train_split=0.6, val_split=0.2, test_split=0.2):
        ''' splits dataset into train, val and test set stratified by class according to proportion \\
            indicated by train_split, val_split and test_split args
            returns: indices correspnding to train and test set
        '''
        if data is None:
            data = self.metadata
        train, val_test = train_test_split(data, train_size=train_split/(train_split+val_split+test_split), random_state=0, stratify=data[['action']])
        val, test = train_test_split(val_test, test_size=test_split/(val_split+test_split), random_state=0, stratify=val_test[['action']])

        train_indices, val_indices, test_indices = self._get_indices(train, val, test)

        assert (len(train_indices) + len(val_indices) + len(test_indices)) == len(self.metadata)
        return train_indices, val_indices, test_indices



class KTHDataset(Dataset):
    """
        KTH action recognition dataset. It contains 599 static videos of 25 actors performing 6 different
        actions in front of 4 different homogeneous backgrounds. (25*6*4 = 600, 1 video is missing)
        The videos are shot on 25 FPS. See more at https://www.csc.kth.se/cvap/actions/.
    """

    joint_indices = {name: i for i, name in enumerate(util.KTH_joint_names)}
    # can access e.g. LKnee for frame (of shape (25,3) by going frame[joint_indices['LKnee'],:]

    def __init__(self, metadata_csv_path, numpy_data_folder, filter=None, transforms=None, use_confidence_scores=True):
        """
        Read and store the metadata of the KTH dataset, i.e. the actor, the label, the scenario and
        the filename of the corresponding numpy data.

        Parameters:
            metadata_csv_path:  Path object of the .csv file which contains the metadata of each video
            numpy_data_folder:  Path object of the folder which contains the joint sequences as .npy files
            filter:  list of indices in metadata_csv file to keep in dataset. If None, entire dataset is kept.
            apply_transforms:  whether to apply transforms to each retrieved point
            use_confidence_scores: if False, trim the OpenPose confidence scores from the returned items
        """
        self.numpy_data_folder = numpy_data_folder
        self.transforms = transforms
        self.use_confidence_scores = use_confidence_scores

        metadata = pd.read_csv(metadata_csv_path)

        if filter is not None:
            metadata = metadata[metadata.index.isin(filter)]
            metadata = metadata.reset_index()

        # TODO Drop the confidence score from OpenPose
        # self.sequences = data['frames'].apply(lambda x: x[:,:,2])

        # convert label names to numbers and store in an ndarray
        self.labels = metadata['action'].apply(lambda x: util.label_name_to_number(x)).to_numpy()
        # append the filenames to the data folder path
        self.filenames = metadata['filename'].apply(lambda x: numpy_data_folder + '/' + x)
        self.actors = metadata['subject']
        self.scenarios = metadata['scenario']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print('[KTHDataset] Using a tensor!')

        # check if slice is given or index
        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            if stop is None: # if end slice not given, go to end of sequence
                stop = len(self.labels)
            if start is None: # if start slice not given, start from beginning of sequence
                start = 0
            idxs = range(start, stop)
        else:
            idxs = [idx]

        # TODO (rajmund): we could add a try-catch here, but I think it would slow us down
        # load sequences and labels for slice/index given
        sequences = np.asarray([np.load(file) for i in idxs for file in [self.filenames[i]]])
        labels = self.labels[idxs]

        # Split the data into (x,y) coordinates and confidence scores (which is the last "coordinate")
        # can't use .split due to different shapes
        # TODO (amrita): change if we save confidence scores into separate .npy files
        joint_sequences = np.array([seq[:, :, :-1] for seq in sequences])

        # TODO (amrita) should we move this outside of class after dataloader has looped so that we can vectorise the matrix multiplication
        # apply data augmentation
        if self.transforms is not None and random.randint(0, 1): # randomly apply augmentation
            joint_sequences = self.transforms(joint_sequences)

        if self.use_confidence_scores:
            print('[ERROR] Confidence scores are not yet supported! Exiting...')
            exit()
            # joint_sequences, confidence_scores = np.split(joint_sequences, [-1], axis=3)
            confidence_scores = np.array([seq[:, :, -1] for seq in joint_sequences])
            return joint_sequences, labels, confidence_scores

        # Otherwise discard the confidence scores
        return joint_sequences, labels

if __name__ == "__main__":
    config = \
    {
        'dataset_dir'   : '../../datasets/KTH_Action_Dataset/',
        'metadata_file' : 'metadata.csv'
    }

    dataset_dir = Path(__file__).parent / config['dataset_dir']
    metadata_file = dataset_dir / config['metadata_file']


    ''' Examine the dataset '''
    # dataset = KTHDataset(metadata_file, dataset_dir, use_confidence_scores=False, apply_transforms=False)
    # print(dataset.filenames[0])
    # seq, action = dataset[0]
    # sequences, action = dataset[:10]
    # print(sequences.shape)

    ''' Train, val, test split'''
    splitDataset = SplitDataset(metadata_file)
    # train_indices, val_indices, test_indices = splitDataset.split_by_scenario(['d1', 'd2'], ['d3'])
    train_indices, val_indices, test_indices = splitDataset.split_by_subject()
    train_dataset = KTHDataset(metadata_file, dataset_dir, filter=train_indices, use_confidence_scores=False, apply_transforms=True)
    val_dataset = KTHDataset(metadata_file, dataset_dir, filter=val_indices, use_confidence_scores=False, apply_transforms=False)
    test_dataset = KTHDataset(metadata_file, dataset_dir, filter=test_indices, use_confidence_scores=False, apply_transforms=False)

    # generate samplers using indices from filtering
    from torch.utils.data import RandomSampler
    train_sampler = RandomSampler(train_dataset)
    val_sampler = RandomSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=10, sampler=train_sampler, collate_fn=util.loopy_pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, collate_fn=util.loopy_pad_collate_fn)



    '''' Batching '''
    for batch_idx, (data, label) in enumerate(train_loader):
        # TODO write a proper test
        print(batch_idx)
        print(label.shape)

        print(data.shape)
        print("Beginning of data 1:")
        print(data[0, :3, 1, :])
        print("Loopy:")
        print(data[0, 360:363, 1, :])

        if batch_idx == len(train_dataset) // 2:
            break


    # loop through train loader and val loader simultaneously
    for batch_idx, ((train_data, train_labels), (val_data, val_labels))in enumerate(zip(train_loader, val_loader)):
        print('Batch', batch_idx)
        print(train_labels.shape)
        print(train_data.shape)
        print(val_labels.shape)
        print(val_data.shape)

