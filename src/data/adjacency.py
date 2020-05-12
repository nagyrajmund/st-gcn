import torch
from enum import IntEnum

from data.util import adj_list, nr_of_joints

class Strategy(IntEnum):
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
        list of partitioned adjacency matrices (matrices are stored as torch.Tensor)
    """

    if strat == Strategy.UNI_LABELING:
        A = torch.zeros((nr_of_joints, nr_of_joints))

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

        I = torch.eye(nr_of_joints)
        matrices = [I]

        # Initialize empty adjacency matrices
        for _ in range(d):
            matrices.append(torch.zeros((nr_of_joints, nr_of_joints)))

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
        matrices:  original adjacency matrices (as list of tensors)
        expo:  exponent (optional)
        alpha:  additional parameter for avoiding empty rows (optional)

    Returns:
        normalized adjacency matrices as a single torch.Tensor
    """

    nr_matrices = len(matrices) # Number of tensors
    first_mat = matrices[0] # Get size of first tensor
    normalized = torch.Tensor(nr_matrices, first_mat.shape[0], first_mat.shape[1])

    for i in range(nr_matrices):
        A = matrices[i]
        Lambda_exp = (torch.diag(torch.sum(A, axis=1)) + alpha) ** expo
        normalized[i, :, :] = Lambda_exp @ A @ Lambda_exp

    return normalized

def get_normalized_adjacency_matrices(strat = Strategy.UNI_LABELING, d = 1, alpha = 0.001):
    """
    Create normalized adjacency matrices.

    Parameters:
        strat:  partitioning strategy (optional)
        d:  distance (optional)
        alpha:  additional parameter for avoiding empty rows (optional)

    Returns:
        normalized adjacency matrices as a single torch.Tensor
    """

    return normalize(create_adjacency_matrices(strat, d), alpha = alpha)