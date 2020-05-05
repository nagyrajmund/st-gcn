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