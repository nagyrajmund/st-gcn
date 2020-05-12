import numpy as np
import os

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_distances(V=25, dataset_dir='../datasets/KTH_Action_Dataset', \
    output_file='../datasets/KTH_Action_Dataset/dist/distances.npy'):
    """
    Calculate average distances from the gravity center.

    Parameters:
        dataset_dir: directory of the data set
        output_file: file to save the distances in (it's best not to put it into dataset_dir)
    """

    # Extract file names
    files = os.listdir(dataset_dir)

    # Vector for storing distances
    distances = np.zeros(V)
    count = np.zeros(V)

    # Iterate through the data sets:
    for f in files:
        if f.endswith(".npy") and f != output_file:
            data = np.load(dataset_dir + "/" + f) #input is in the form (T, V, 3)

            T = data.shape[0]

            # Frames
            for t in range(T):
                x = data[t, :, 0]
                y = data[t, :, 1]

                # Gravity center
                grav = (np.average(x), np.average(y))

                for i in range(V):
                    # Current joint
                    joint = (data[t, i, 0], data[t, i, 1])
                    dist = calculate_distance(grav, joint)
                    distances[i] += dist
                    count[i] += 1 # Keep count in case not every node is present 

    distances = distances/count

    np.save(output_file, distances)