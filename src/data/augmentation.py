import numpy as np
import os, subprocess
from pathlib import Path
from util import connections
from shutil import rmtree
import matplotlib.pyplot as plt
from datasets import KTHDataset

def _convert_images_to_video(images_folder, output_vid_path):
    '''
    Converts images in images_folder to video and saves as .mp4.

    images_folder: path to folder containing images to convert to mp4
    output_vid_path: path to save output video

    returns: None
    '''

    print(f'Converting images in {images_folder} and saving to video {output_vid_path}...')
    fps = 30 # default frames per second in openpose

    # ffmpeg command to convert images to video
    command = f'ffmpeg -framerate {fps} -i {images_folder}/%d.png -c:v libx264 -pix_fmt yuv420p {output_vid_path}'
    commands = command.split(' ')
    subprocess.call(commands)

    print('Done')


def plot_skeleton(seq, output_fpath, vidname=None):
    '''
        Plots skeleton from keypoints in .json files for video
    Inputs:
        seq: array of keypoints for frames corresponding to video to plot (N_frames, N_joints, 3)
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


def augment_data(seq):
    '''
    Applies random moving to sequence of shape (N_frames, 25, 3)
    '''
    # TODO change to batch on sequences. Or do it on the fly in torch DataLoader?

    seq_copy = np.copy(seq)
    transformed_frames = seq_copy[:,:,:2] # TODO remove once removed confidence scores @amrita
    N_frames, N_joints, d = seq_copy.shape

    # here x axis points right, y axis points up
    # TODO come up with more sensible decisions for these pre-chosen parameters
    rotations = [15, -15, 5, -5, 10, -10] # each sublist is a set of rotations about origin
    translation = [[5, 5], [0, 5], [5, 0]] # each sublist is a set of translations on x axis, y axis
    scale_factors = [1.05, 1.1, 0.95]

    # choose 3 out of 4 transformations to apply
    transformations = ['rotation', 'translation', 'scaling', 'flip']
    # transformations_to_apply = np.random.choice(transformations, 3)
    transformations_to_apply = transformations
    # apply rotation
    if 'rotation' in transformations_to_apply:
        print('rotating')
        # randomly select rotation
        theta = np.radians(np.random.choice(rotations))
        c, s = np.cos(theta), np.sin(theta)
        rot_matx = np.array([[c, s], [-s, c]])
        # flatten to rotate across frames
        transformed_frames = np.reshape(transformed_frames, (N_frames*N_joints, d))
        transformed_frames = np.dot(transformed_frames, rot_matx)
        # reshape back
        transformed_frames = np.reshape(transformed_frames, (N_frames, N_joints, d))

    # apply translation
    if 'translation' in transformations_to_apply:
        print('translating')
        t = translation[np.random.choice(range(3))]
        transformed_frames += t

    # apply scaling factor
    if 'scaling' in transformations_to_apply:
        print('scaling')
        scale_factor = np.random.choice(scale_factors)
        transformed_frames *= scale_factor

    if 'flip' in transformations_to_apply:
        transformed_frames[:,:,0] = -transformed_frames[:,:,0]
    # TODO interpolate transformation during frames in sequence to generate smooth effect of 'random moving'? @amrita

    seq_copy[:,:,:2] = transformed_frames
    return transformed_frames


if __name__ == '__main__':
    # read in sequence
    dataset_dir = Path(__file__).parent / '../../datasets/KTH_Action_Dataset/'
    metadata_file = dataset_dir / 'metadata.csv'
    a = KTHDataset(metadata_file, dataset_dir)
    print(a.filenames[0])
    seq, action, scores = a[0]
    print(seq[0].shape)
    # plot
    augmented_seq = augment_data(seq[0])
    plot_skeleton(augmented_seq, '../../datasets/example_augmented_plot.mp4')
