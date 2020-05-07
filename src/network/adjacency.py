import numpy as np
from enum import Enum

from data.util import adj_list, nr_of_joints

class Strategy(Enum):
    UNI_LABELING = 0
    DISTANCE = 1
    SPATIAL_CONFIGURATION = 2

def create_adjacency_matrices(strat = Strategy.UNI_LABELING, d = 1):
    """
    Create adjacency matrices.

    Parameters:
        strat:  partitioning strategy
        d:  distance

    Returns:
        list of adjacency matrices (partitioned)
    """

    if strat == Strategy.UNI_LABELING:
        A = np.zeros((nr_of_joints, nr_of_joints), dtype=int) # TODO turn it into boolean array?

        for i in range(nr_of_joints):
            # For each joint, do a (limited) DFS.
            neighbour_elements = [i] # Contains every element in the neighbourhood
            open_elements = [i] # Contains only those that belong to the most recent iteration

            for _ in range(d):
                new_open = []
                while open_elements:
                    curr = open_elements.pop(0)
                    new_elements = [x for x in adj_list[curr] if x not in neighbour_elements]
                    neighbour_elements.extend(new_elements)
                    new_open.extend(new_elements)

                open_elements = new_open

            # Fill corresponding row.
            for neighbour in neighbour_elements:
                A[i][neighbour] = 1

        return [A]

    elif strat == Strategy.DISTANCE:

        I = np.eye(nr_of_joints, dtype=int) # TODO turn it into boolean array?
        matrices = [I]

        for _ in range(d):
            A = np.zeros((nr_of_joints, nr_of_joints), dtype=int) # TODO turn it into boolean array?
            matrices.append(A)

        for i in range(nr_of_joints):
            # For each joint, do a (limited) DFS.
            neighbour_elements = [i] # Contains every element in the neighbourhood
            open_elements = [i] # Contains only those that belong to the most recent iteration

            for dist in range(d):
                new_open = []
                while open_elements:
                    curr = open_elements.pop(0)
                    new_elements = [x for x in adj_list[curr] if x not in neighbour_elements]
                    neighbour_elements.extend(new_elements)
                    new_open.extend(new_elements)

                open_elements = new_open
                for neighbour in open_elements:
                    matrices[dist + 1][i][neighbour] = 1

        return matrices

    elif strat == Strategy.SPATIAL_CONFIGURATION:
        # TODO implement. It needs the input or at least the distances from the gravity center
        raise NotImplementedError()

        matrices = []
        return matrices

def normalize(matrices, expo=-1/2, alpha = 0.001):
    """
    Normalize adjacency matrices.

    Parameters:
        matrices:  original adjacency matrices
        expo:  exponent (optional)
        alpha:  additional parameter for avoiding empty rows (optional)

    Returns:
        normalized adjacency matrices as a numpy array
    """

    normalized = []
    for A in matrices:
        Lambda_exp = (np.diag(np.sum(A, axis=1)) + alpha) ** expo
        normalized.append(Lambda_exp @ A @ Lambda_exp)
    return np.asarray(normalized)

def get_normalized_adjacency_matrices(strat = Strategy.UNI_LABELING, d = 1, alpha = 0.001):
    """
    Create normalized adjacency matrices.

    Parameters:
        strat:  partitioning strategy (optional)
        d:  distance (optional)
        alpha:  additional parameter for avoiding empty rows (optional)
    """

    return normalize(create_adjacency_matrices(strat, d), alpha = alpha)