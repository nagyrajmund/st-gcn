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


'''
Edge list for storing joint connections.
'''
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

