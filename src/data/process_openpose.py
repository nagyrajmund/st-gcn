import numpy as np
import os
import json
from pathlib import Path
import pandas as pd


actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


def process_openpose(dataset_dir, output_dir='../datasets/KTH_Action_Dataset'):
    ''' Processes json keypoints for actions in dataset_dir and stores in a npy file per video '''

    metadata = {'subject': [], 'action': [], 'scenario': [], 'filename': []}

    for action in actions:
        print('action', action)
        vidnames = os.listdir(dataset_dir + '/' + action)
        vidnames = [fname[:-28] for fname in vidnames] # just retrieve videoname stem
        vidnames = list(set(vidnames))
        vidnames.sort()

        action_path = Path(dataset_dir + '/' + action)
        for vidname in vidnames:
            skeletons = []  # array [[x0, y1, x1, y1, x2, y2,...] for each frame]
            vid_json_files = list(action_path.glob(vidname+'*.json'))

            # loop over all json files for current video
            for i, json_path in enumerate(vid_json_files):
                data = json.load(open(str(json_path)))
                # take first person
                try:
                    person = data['people'][0]
                    frame_coords_scores = np.array(person['pose_keypoints_2d']).reshape(25,3) # stores [[x_n,y_n,c_n] for each joint n ] for current frame
                    skeletons.append(frame_coords_scores)
                except:
                    print('error: no person found in frame for vid',  vidname, '. Ignoring frame', i)
                    continue

            subject, _, scenario, _ = vidname.split('_')

            # save skeletons for video to .npy file
            npy_fpath = f'{output_dir}/{subject}_{action}_{scenario}.npy'
            skeletons = np.asarray(skeletons)
            np.save(npy_fpath, skeletons)

            # add data for video to dataset dictionary
            metadata['subject'].append(subject)
            metadata['action'].append(action)
            metadata['scenario'].append(scenario)
            metadata['filename'].append(npy_fpath)


    metadata_fpath = f'{output_dir}/metadata.csv'
    print('Total number of sequences', len(metadata['subject']))
    print('Saving metadata to', metadata_fpath, '...')
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_fpath)
    print('Done.')


def check_all_videos_processed(videos_dir, dataset_dir):
    ''' checks all videos have been processed by openpose and are saved in json keypoints '''
    count = 0
    for action in actions:
        print('Checking all videos have been converted by openpose for action', action)
        vidnames = os.listdir(videos_dir + '/' + action)
        vidnames = [vidname[:-4] for vidname in vidnames] # remove .avi
        vidnames.sort()
        keypoints_vidnames = os.listdir(dataset_dir + '/' + action)
        keypoints_vidnames = list(set([fname[:-28] for fname in keypoints_vidnames]))
        keypoints_vidnames.sort()

        assert vidnames == keypoints_vidnames
        count += len(vidnames)
        print('Done.')
    print(count) # 599
    # note authors say there are 600 (100/action) but the handclapping action actually only has 99 videos so total # videos = 599 '''


def to_reprocess(dataset_dir):
    ''' checks which videos are missing more than 10 frames in a row and should be reprocessed '''

    redo = []
    for action in actions:
        print('action', action)
        vidnames = os.listdir(dataset_dir + '/' + action)
        vidnames = [fname[:-28] for fname in vidnames] # just retrieve videoname stem
        vidnames = list(set(vidnames))
        vidnames.sort()

        action_path = Path(dataset_dir + '/' + action)
        for vidname in vidnames:
            vid_json_files = list(action_path.glob(vidname+'*.json'))
            vid_json_files.sort()

            # loop over all json files for current video
            counts = []
            for i, json_path in enumerate(vid_json_files):
                data = json.load(open(str(json_path)))
                # take first person
                try:
                    person = data['people'][0]
                    frame_coords_scores = np.array(person['pose_keypoints_2d'])
                except:
                    assert data['people'] == []
                    if counts == []:
                        counts.append(i)
                    elif counts[-1] == i-1:
                        counts.append(i)
                    else:
                        counts = []

                    if len(counts) >= 30:
                        redo.append(vidname)
                        counts = []

                    continue

    # videos with more than 5 frames in a row missing during sequence
    print(list(set(redo)))



if __name__ == '__main__':
    # check_all_videos_processed('../KTH_Action_Dataset/videos', '../KTH_Action_Dataset/keypoints')
    # to_reprocess('../KTH_Action_Dataset/keypoints')

    process_openpose('../KTH_Action_Dataset/keypoints')

