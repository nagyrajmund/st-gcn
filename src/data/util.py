import sys
sys.path.append('./')

import numpy as np
import torch
import os, subprocess
import matplotlib.pyplot as plt
from shutil import rmtree

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


def _convert_images_to_video(images_folder, output_vid_path):
    '''
    Converts images in images_folder to video and saves as .mp4.

    Parameters:
        images_folder:  path to folder containing images to convert to mp4
        output_vid_path:  path to save output video

    Returns:
        None
    '''

    print(f'Converting images in {images_folder} and saving to video {output_vid_path}...')
    fps = 30 # default frames per second in openpose

    # ffmpeg command to convert images to video
    command = f'ffmpeg -framerate {fps} -i {images_folder}/%d.png -c:v libx264 -pix_fmt yuv420p -vf pad=ceil(iw/2)*2:ceil(ih/2)*2 {output_vid_path}'
    commands = command.split(' ')
    subprocess.call(commands)

    print('Done')


def plot_skeleton(seq, output_fpath):
    '''
        Plots skeleton from keypoints in .json files for video.

    Inputs:
        seq: array of keypoints for frames corresponding to video to plot (N_frames, N_joints, 2)
        output_fpath: fpath to save output video with overlay of openpose keypoints

    Returns:
        None
    '''
    # TODO remove extra point being plotted in frame from extra irrelevant joint @amrita

    assert seq.shape[1] == 25
    N_frames = len(seq)
    output_dir = os.path.dirname(output_fpath)
    assert os.path.isdir(output_dir)

    output_tmp_dir = output_dir + '/tmp' # temp directory to store images plotted per frame
    if os.path.isdir(output_tmp_dir):
        print('Plotting to a directory that possibly already has images in it')
    else:
        os.mkdir(output_tmp_dir)

    for i in range(N_frames):
        skeleton = np.array(seq[i])  # get skeleton at frame i of video
        x = skeleton[:, 0]
        y = skeleton[:, 1] # reverse vertical axis for plotting purposes

        fig, ax = plt.subplots(1, figsize=(3, 8))
        sc = ax.scatter(x, -y, s=40)
        plt.gca().set_aspect('equal', adjustable='box')

        for bone in connections:
            x0 = x[bone[0]]
            y0 = y[bone[0]]
            x1 = x[bone[1]]
            y1 = y[bone[1]]
            if not ([x0, y0] == [0, 0] or [x1, y1] == [0, 0]):  # don't plot if one of joints is 0 i.e. missing estimate
                ax.plot([x[bone[0]], x[bone[1]]], [-y[bone[0]], -y[bone[1]]], 'g')

        plt.axis('off')
        plt.savefig(f'{output_tmp_dir}/{i}.png', bbox_inches='tight')
        plt.close(fig)

    _convert_images_to_video(output_tmp_dir, output_fpath)
    # remove tmp directory of images
    rmtree(output_tmp_dir)
