import sys
sys.path.append('../')

import numpy as np
from pathlib import Path


def augment_data(sequences):
    '''
    Applies random moving to sequences each of shape (N_frames, 25, 2).

    Parameters:
        sequences:  sequences
    '''
    rotations = [15, -15, 5, -5, 10, -10] # each sublist is a set of rotations about origin
    translation = [[5, 5], [0, 5], [5, 0]] # each sublist is a set of translations on x axis, y axis
    scale_factors = [1.05, 1.1, 0.95]

    # choose 3 out of 4 transformations to apply
    transformations = ['rotation', 'translation', 'scaling', 'flip']
    transformations_to_apply = np.random.choice(transformations, 2)
    T = np.eye((3))


    # apply rotation
    if 'rotation' in transformations_to_apply:
        theta = np.radians(np.random.choice(rotations)) # randomly select rotation to apply
        c, s = np.cos(theta), np.sin(theta)
        rot_matx = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T = rot_matx.dot(T)

    # apply translation
    if 'translation' in transformations_to_apply:
        t_x, t_y = translation[np.random.choice(range(3))] # randomly select translation to apply
        t_matx = np.array([[1,0,t_x], [0,1,t_y], [0,0,1]])
        T = t_matx.dot(T)

    # apply scaling factor
    if 'scaling' in transformations_to_apply:
        scale_factor = np.random.choice(scale_factors) # randomly select scaling to apply
        scale_mtx = np.array([[scale_factor,0,0], [0,scale_factor,0], [0,0,1]])
        T = scale_mtx.dot(T)

    if 'flip' in transformations_to_apply:
        flip_mtx = np.array(([[-1,0,0],[0,1,0],[0,0,1]]))
        T = flip_mtx.dot(T)

    # TODO interpolate transformation during frames in sequence to generate smooth effect of 'random moving' @amrita
    # TODO move to Data class

    def _transform_sequence(x):
        N_frames, N_joints, d = x.shape
        assert N_joints == 25

        transformed_frames = np.zeros((N_frames, N_joints, d+1))
        transformed_frames[:,:,:2] = np.copy(x)

        transformed_frames = np.reshape(transformed_frames, (N_frames * N_joints, d+1)) # tODOcome back to
        transformed_frames = np.dot(transformed_frames, T)

        # reshape back
        transformed_frames = np.reshape(transformed_frames, (N_frames, N_joints, d+1))

        return transformed_frames[:,:,:2]

    # apply T
    transformed_sequences = np.asarray([_transform_sequence(seq) for seq in sequences])

    return transformed_sequences



if __name__ == '__main__':
    # read in sequence
    dataset_dir = Path(__file__).parent / '../../datasets/KTH_Action_Dataset/'
    metadata_file = dataset_dir / 'metadata.csv'
    from datasets import KTHDataset
    dataset = KTHDataset(metadata_file, dataset_dir, use_confidence_scores=False)
    seqs, actions = dataset[:4]
    original_seq = np.copy(seqs[0])

    augmented_sequences = augment_data(seqs[:4])

    assert np.array_equal(seqs[0], original_seq) # check original sequence still intact

    from util import plot_skeleton
    print('Saving plot of skeleton')
    plot_skeleton(augmented_sequences[0], '../../datasets/example_augmented_plot.mp4')
