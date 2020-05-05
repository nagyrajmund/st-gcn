import pandas as pd
from ast import literal_eval
import numpy as np
from torch.utils.data import Dataset
import util
import torch
from pathlib import Path

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
        
        # TODO (rajmund): we could add a try-catch here, but I think it would slow us down
        # TODO (rajmund): make this work for both slices and single indices
        labels = self.labels[idx]
        joint_sequences = np.array([np.load(file) for file in [self.filenames[idx]]])
        
        if self.use_confidence_scores:
            # Split the data into (x,y) coordinates and confidence scores (which is the last "coordinate") 
            joint_sequences, confidence_scores = np.split(joint_sequences, [-1], axis=3)

            return joint_sequences, labels, confidence_scores
        
        # Otherwise discard the confidence scores
        return joint_sequences[:,:,:-1], labels

if __name__ == "__main__":
    dataset_dir = Path(__file__).parent / '../../datasets/KTH_Action_Dataset/'
    metadata_file = dataset_dir / 'metadata.csv'
    a = KTHDataset(metadata_file, dataset_dir)
    print(a.filenames[0])
    seq, action, scores = a[150]    
    print(seq.shape, action, scores.shape)