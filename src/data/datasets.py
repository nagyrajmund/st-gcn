# TODO: this should be added to every runnable script to make imports work
# Eventually we can use python 
import sys
sys.path.append('./')
import pandas as pd

import numpy as np
from torch.utils.data import Dataset
import util
import torch
from pathlib import Path
from torch.utils.data import DataLoader

class KTHDataset(Dataset):
    """
        KTH action recognition dataset. It contains 599 static videos of 25 actors performing 6 different
        actions in front of 4 different homogeneous backgrounds. (25*6*4 = 600, 1 video is missing)
        The videos are shot on 25 FPS. See more at https://www.csc.kth.se/cvap/actions/.
    """

    joint_indices = {name: i for i, name in enumerate(util.KTH_joint_names)}
    # can access e.g. LKnee for frame (of shape (25,3) by going frame[joint_indices['LKnee'],:]

    def __init__(self, metadata_csv_path, numpy_data_folder, transforms = None, use_confidence_scores=True):
        """
        Read and store the metadata of the KTH dataset, i.e. the actor, the label, the scenario and
        the filename of the corresponding numpy data.

        Parameters:
            metadata_csv_file:  Path object of the .csv file which contains the metadata of each video
            numpy_data_folder:  Path object of the folder which contains the joint sequences as .npy files
            transforms:  Transformations to apply to each retrieved datapoint
            use_confidence_scores: If False, trim the OpenPose confidence scores from the returned items
        """
        self.numpy_data_folder = numpy_data_folder
        self.transforms = transforms
        self.use_confidence_scores = use_confidence_scores

        metadata = pd.read_csv(metadata_csv_path)
        # # Drop the confidence score from OpenPose
        # self.sequences = data['frames'].apply(lambda x: x[:,:,2])

        # convert label names to numbers and store in an ndarray
        self.labels = metadata['action'].apply(lambda x: util.label_name_to_number(x)).to_numpy()
        # append the filenames to the data folder path
        self.filenames = metadata['filename'].apply(lambda x: numpy_data_folder / x)
        self.actors = metadata['subject']
        self.scenarios = metadata['scenario']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print('here')

        # TODO (rajmund): we could add a try-catch here, but I think it would slow us down
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

        # load sequences and labels for slice/index given
        sequences = np.asarray([np.load(file) for i in idxs for file in [self.filenames[i]]])
        labels = self.labels[idxs]

        # Split the data into (x,y) coordinates and confidence scores (which is the last "coordinate")
        # can't use .split due to different shapes
        # TODO (amrita): change if we save confidence scores into separate .npy files
        joint_sequences = np.array([seq[:, :, :-1] for seq in sequences])

        if self.use_confidence_scores:
            # joint_sequences, confidence_scores = np.split(joint_sequences, [-1], axis=3)
            confidence_scores = np.array([seq[:, :, -1] for seq in joint_sequences])
            return joint_sequences, labels, confidence_scores

        # Otherwise discard the confidence scores
        return joint_sequences, labels



if __name__ == "__main__":
    dataset_dir = Path(__file__).parent / '../../datasets/KTH_Action_Dataset/'
    metadata_file = dataset_dir / 'metadata.csv'
    a = KTHDataset(metadata_file, dataset_dir)
    # print(a.filenames[0])
    seq, action, scores = a[0]
    sequences, action, scores = a[:10]

    print(sequences.shape)

    dataloader = DataLoader(a, 10, sampler=None)

    # for batch_idx, (data, label, scores) in enumerate(dataloader):
    #     TODO problem with batch size > 1
    #     print(batch_idx)
    #     print(data.shape)
    #     input()




