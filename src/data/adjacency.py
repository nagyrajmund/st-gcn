import torch
import numpy as np
from enum import IntEnum

from data.util import adj_list, nr_of_joints

class Strategy(IntEnum):
    UNI_LABELING = 0
    DISTANCE = 1
    SPATIAL_CONFIGURATION = 2

def increase_neighbourhood(neighbour_elements, open_elements):
    """
    Simulates a step in DFS.

    Parameters:
        neighbour_elements:  nodes that are considered to be in the neighbourhood so far
        open_elements:  open nodes; nodes that have been visited but not processed

    Returns:
        new_open:  new open nodes
    """
    new_open = []

    while open_elements:
        curr = open_elements.pop(0)
        new_elements = [x for x in adj_list[curr] if x not in neighbour_elements]
        neighbour_elements.extend(new_elements)
        new_open.extend(new_elements)

    return new_open

def create_adjacency_matrices(strat = Strategy.UNI_LABELING, d = 1, distance_file = None):
    """
    Create adjacency matrices.

    Parameters:
        strat:  partitioning strategy
        d:  distance
        distance_file:  file that store the average distances from the gravity center (spatial conf. partitioning)

    Returns:
        list of partitioned adjacency matrices (matrices are stored as torch.Tensor)
    """

    ################
    # Uni-labeling #
    ################
    if strat == Strategy.UNI_LABELING:
        A = torch.zeros((nr_of_joints, nr_of_joints))

        # For each joint, do a (limited) DFS.
        for i in range(nr_of_joints):
            neighbour_elements = [i] # Contains every element in the neighbourhood
            open_elements = [i] # Contains only those that belong to the most recent iteration

            for _ in range(d):
                open_elements = increase_neighbourhood(neighbour_elements, open_elements)

            # Fill corresponding row.
            for neighbour in neighbour_elements:
                A[i][neighbour] = 1

        return [A]

    #########################
    # Distance partitioning #
    #########################
    elif strat == Strategy.DISTANCE:

        I = torch.eye(nr_of_joints)
        matrices = [I]

        # Initialize empty adjacency matrices
        for _ in range(d):
            matrices.append(torch.zeros((nr_of_joints, nr_of_joints)))

        # For each joint, do a (limited) DFS.
        for i in range(nr_of_joints):
            neighbour_elements = [i] # Contains every element in the neighbourhood
            open_elements = [i] # Contains only those that belong to the most recent iteration

            for dist in range(d):
                open_elements = increase_neighbourhood(neighbour_elements, open_elements)

                for neighbour in open_elements:
                    matrices[dist + 1][i][neighbour] = 1

        return matrices

    ######################################
    # Spatial configuration partitioning #
    ######################################
    elif strat == Strategy.SPATIAL_CONFIGURATION:
        nr_of_partitions = 3

        # This needs the distances from the gravity center
        if distance_file is None:
            raise ValueError("Distance file not provided")
        avg_distances = np.load(distance_file) # Distances for each node stored in a numpy array (vector consisting of V elements)

        # Initialize empty adjacency matrices
        matrices = []
        for _ in range(nr_of_partitions):
            matrices.append(torch.zeros((nr_of_joints, nr_of_joints)))

        # For each joint, do a (limited) DFS.
        for i in range(nr_of_joints):
            neighbour_elements = [i] # Contains every element in the neighbourhood
            open_elements = [i] # Contains only those that belong to the most recent iteration

            for _ in range(d):
                open_elements = increase_neighbourhood(neighbour_elements, open_elements)

            # Fill corresponding row by comparing the distances of the neighbouring nodes from the center.
            root_distance = avg_distances[i]
            for neighbour in neighbour_elements:
                neighbour_distance = avg_distances[neighbour]
                if neighbour_distance == root_distance:
                    label = 0
                elif neighbour_distance < root_distance:
                    label = 1
                else:
                    label = 2
                matrices[label][i][neighbour] = 1

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

    nr_matrices = len(matrices) # Number of tensors (== number of partitions)
    first_mat = matrices[0] # Get size of first tensor
    normalized = torch.Tensor(nr_matrices, first_mat.shape[0], first_mat.shape[1])

    for i in range(nr_matrices):
        A = matrices[i]
        Lambda_exp = (torch.diag(torch.sum(A, axis=1)) + alpha) ** expo
        normalized[i, :, :] = Lambda_exp @ A @ Lambda_exp

    return normalized

def get_normalized_adjacency_matrices(strat = Strategy.UNI_LABELING, d = 1, alpha = 0.001, distance_file = None):
    """
    Create normalized adjacency matrices.

    Parameters:
        strat:  partitioning strategy (optional)
        d:  distance (optional)
        alpha:  additional parameter for avoiding empty rows (optional)
        distance_file:  file that store the average distances from the gravity center (spatial conf. partitioning)

    Returns:
        normalized adjacency matrices as a single torch.Tensor
    """

    return normalize(create_adjacency_matrices(strat, d, distance_file = distance_file), alpha = alpha)