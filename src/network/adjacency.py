import numpy as np
from enum import Enum

from data.util import connections

'''
Adjacency list.
'''
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

nr_of_joints = 25

class Strategy(Enum):
    UNI_LABELING = 0
    DISTANCE = 1
    SPATIAL_CONFIGURATION = 2

def create_adjacency_matrices(strat = Strategy.UNI_LABELING, d = 1):
    """
    Create adjacency matrices.

    strat: partitioning strategy
    d: distance

    returns: list of adjacency matrices (partitioned)
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
        # TODO implement
        raise NotImplementedError()

        matrices = []
        return matrices

def normalize(matrices, expo=-1/2, alpha = 0.001):
    normalized = []
    for A in matrices:
        Lambda_exp = (np.diag(np.sum(A, axis=1)) + alpha) ** expo
        normalized.append(Lambda_exp @ A @ Lambda_exp)
    return normalized
