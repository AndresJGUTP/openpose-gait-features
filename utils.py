import os
import glob
import json
import numpy as np
from model25_keypoints import Model25_KP

def from_json(path):
    file = open(path, 'r', encoding='utf-8')
    return json.load(file)

def to_json(data, path):
    with open(path, "w") as write_file:
        json.dump(data, write_file, indent=4)

def extract_openpose_anns(ann_json):
    def extract_keypoints(ann_json):
        X = []
        Y = []
        C = []
        id = 0
        while id < len(ann_json):
            X.append(ann_json[id])
            Y.append(ann_json[id+1])
            C.append(ann_json[id+2])
            id += 3

        return np.array([X, Y, C])

    pose = {}
    # If there aren't people in frame, return zeros
    if len(ann_json['people']) > 0:
        kp_pose = extract_keypoints(ann_json['people'][0]['pose_keypoints_2d'])
        kp_face = extract_keypoints(ann_json['people'][0]['face_keypoints_2d'])
        kp_hand_left = extract_keypoints(ann_json['people'][0]['hand_left_keypoints_2d'])
        kp_hand_right = extract_keypoints(ann_json['people'][0]['hand_right_keypoints_2d'])

        pose['pose'] = kp_pose
        pose['face'] = kp_face
        pose['hand_left'] = kp_hand_left
        pose['hand_right'] = kp_hand_right
    else:
        X = [0] * 25
        Y = [0] * 25
        C = [0] * 25
        pose['pose'] = [X, Y, C]
        # ToDo: Fix for this keypoints
        pose['face'] = [[], [], []]
        pose['hand_left'] = [[], [], []]
        pose['hand_right'] = [[], [], []]

    return pose

def get_pose_annotations(path):
    '''
    Retorna la trayectoria temporal (por cada frame) de los keypoints
    -> lista_frames(total frames) - dict_keypoints(pose, hands, face) - lista_keypoints(model_25)
    '''
    path = os.path.join(path,'*')
    files = glob.glob(path)
    files.sort()

    Y_raw = []
    for file in files:
        ann_json = from_json(file)
        ann = extract_openpose_anns(ann_json)
        Y_raw.append(ann)

    return Y_raw

def get_keypoint_trajectory(keypoint, Y):
    X_trajectory = []
    Y_trajectory = []
    C_trajectory = []
    for frame in Y:
        pose_keypoints = frame['pose']
        X_trajectory.append( pose_keypoints[0][Model25_KP[keypoint].value] )
        Y_trajectory.append( pose_keypoints[1][Model25_KP[keypoint].value] )
        C_trajectory.append( pose_keypoints[2][Model25_KP[keypoint].value] )

    return {'x': X_trajectory, 'y': Y_trajectory, 'c': C_trajectory}
    # return np.array([ X_trajectory, Y_trajectory, C_trajectory ])

def get_all_keypoints_trajectory(trajectories_by_frame):
    trajectories = {}
    for keypoint in Model25_KP:
        trajectories[keypoint.name] = get_keypoint_trajectory(keypoint.name, trajectories_by_frame)

    return trajectories

def get_body_edges(joint):
    edges = []
    if joint == Model25_KP.RKnee.name:
        edges = [(10, 11), (9, 10)]
    elif joint == Model25_KP.LKnee.name:
        edges = [(13, 14), (12, 13)]
    elif joint == Model25_KP.RHip.name:
        edges = [(10, 9), (1, 8)]
    elif joint == Model25_KP.LHip.name:
        edges = [(13, 12), (1, 8)]
    elif joint == Model25_KP.RAnkle.name:
        edges = [(10, 11), (11, 22)]
    elif joint == Model25_KP.LAnkle.name:
        edges = [(13, 14), (14, 19)]
    

    else:
        assert False, 'Invalid body part'

    return edges