import sys
sys.path.append('./')

import numpy as np
import torch

### Helpers for handling dataset itself

def pad_array_with_loops(x, target_len):
    """Pad an array on the second axis to the specified length by 
    looping its values from the beginning.
    
    Parameters:
        x:  a 4-dimensional array of shape (N,T,V,C)
        target_len:  the desired length
    
    Returns:  The padded array of shape (N,target_len,V,C)
    """
    x_len = x.shape[1] # on time axis
    if x_len >= target_len:
        return x
    
    # For info about pad_widths, visit the official documentation for numpy.pad().
    pad_widths = [(0,0), (0, target_len - x_len), (0,0), (0,0)]
    

    return np.pad(x, pad_widths, mode='wrap')


def loopy_pad_collate_fn(batch):
    """Pad each sequence to the maximum length in the batch along the time axis, then combine them into a tensor.
    
    Parameters:
        batch:  A list that has batch_size * tuple(f_in, label) elements, where
                f_in is the (N, T, V, C) skeleton sequence array and label is the corresponding label. 
    Returns:  
        xx:  A tensor of shape (N,T*,V,C) where T* is the length of the longest sequence in batch.
        labels:  A tensor of shape (N).
    """ 
    max_len = max([x[0].shape[1] for x in batch])
    xx = [torch.from_numpy(pad_array_with_loops(x[0], max_len)) for x in batch]
    labels = [torch.from_numpy(x[1]) for x in batch]

    return torch.cat(xx), torch.cat(labels)

### Constants and enums/transformations related to data representation.
nr_of_joints = 25

KTH_label_name_to_number = \
    {"boxing"      : 0,
    "handclapping" : 1,
    "handwaving"   : 2,
    "jogging"      : 3,
    "running"      : 4,
    "walking"      : 5}

def label_name_to_number(label_name):
    return KTH_label_name_to_number[label_name]

# All the joints in the KTH dataset.
KTH_joint_names = \
    ["Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    "Background"]

# Edge list for storing joint connections.
connections = [(0, 1),
              (1, 2),
              (2, 3),
              (3, 4),
              (1, 5),
              (5, 6),
              (6, 7),
              (1, 8),
              (8, 9),
              (9, 10),
              (10, 11),
              (8, 12),
              (12, 13),
              (13, 14),
              (0, 15),
              (0, 16),
              (15, 17),
              (16, 18),
              (14, 19),
              (19, 20),
              (14, 21),
              (11, 22),
              (22, 23),
              (11, 24)]

# Adjacency list.
adj_list = {0: [1, 15, 16],
         1: [0, 2, 5, 8],
         2: [1, 3],
         3: [2, 4],
         4: [3],
         5: [1, 6],
         6: [5, 7],
         7: [6],
         8: [1, 9, 12],
         9: [8, 10],
         10: [9, 11],
         11: [10, 22, 24],
         12: [8, 13],
         13: [12, 14],
         14: [13, 19, 21],
         15: [0, 17],
         16: [0, 18],
         17: [15],
         18: [16],
         19: [14, 20],
         20: [19],
         21: [14],
         22: [11, 23],
         23: [22],
         24: [11]}
