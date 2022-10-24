import os
from utils import *
from model25_keypoints import Model25_KP
import json
import constants
from constants import edges, JOINT_ANGLES, RIGHT_COLOR, LEFT_COLOR
import matplotlib.pyplot as plt
import numpy as np
import math

class Config_OpenPose():
    def __init__(self):
        self.format = 'avi'

class OPEngine():
    def __init__(self, subject_id, base_path=""):
        self.subject_id = subject_id
        self.base_path = base_path
        self.cfg_openpose = Config_OpenPose()
        self.cfg_engine = {}

        # Dirs
        self.save_kp_video_path = os.path.join(base_path, f'{self.subject_id}_keypoints.{self.cfg_openpose.format}')
        self.save_kp_frames_json_path = os.path.join(base_path, subject_id, 'json_frames')
        self.save_config_path = os.path.join(base_path, subject_id, 'OPEngine_config')
        self.save_figures = os.path.join(base_path, subject_id, 'figures')
        self.save_data = os.path.join(base_path, subject_id, 'data')
        
        # Create dirs
        self.createFolder(self.base_path)
        self.createFolder(os.path.join(self.base_path, self.subject_id))
        self.createFolder(self.save_figures)
        self.createFolder(self.save_data)

    def build_folder_structure(self):
        self.createFolder(self.save_kp_frames_json_path)
        self.createFolder(self.save_config_path)
    
    def createFolder(self, new_folder, clean=False):
        if os.path.isdir(new_folder):
            if clean:
                os.rmdir(new_folder)
                os.mkdir(new_folder)
            else:
                print(f'Folder: {new_folder} already exists')
        else:
            os.mkdir(new_folder)

    #TODO: Take a video from input path and get keypoints using openpose and save them
    def proccess_openpose(self, input_path_video):
        pass

    def proccess_trajectories(self, save=True):
        Y_trajectories_by_frame = get_pose_annotations(self.save_kp_frames_json_path)
        trajectories_by_kp = get_all_keypoints_trajectory(Y_trajectories_by_frame)

        if save:
            path = os.path.join(self.save_data, 'raw_keypoints_trajectories.json')
            to_json(trajectories_by_kp, path)
        
        return trajectories_by_kp

    def load_kp_trajectories(self, filename):
        path = os.path.join(self.save_data, filename)
        kp_trajectories = from_json(path)
        
        for key in kp_trajectories.keys():
            kp_trajectories[key]['x'] = np.array(kp_trajectories[key]['x'])
            kp_trajectories[key]['y'] = np.array(kp_trajectories[key]['y'])
        
        return kp_trajectories

    def fill_nan_values(self, data):
        ok = data != 0
        if ok.any():
            xp = ok.ravel().nonzero()[0]
            fp = data[ok]
            x  = (data == 0).ravel().nonzero()[0]
            data[data == 0] = np.interp(x, xp, fp)
        return data

    def handle_missing_keypoints(self, kp_trajectories, save=True):
        for keypoint in Model25_KP:
            kp_trajectories[keypoint.name]['x'] = self.fill_nan_values(kp_trajectories[keypoint.name]['x'])
            kp_trajectories[keypoint.name]['y'] = self.fill_nan_values(kp_trajectories[keypoint.name]['y'])

        kp_trajectories_copy = kp_trajectories.copy()

        if save:
            for keypoint in Model25_KP:
                kp_trajectories_copy[keypoint.name]['x'] = kp_trajectories_copy[keypoint.name]['x'].tolist()
                kp_trajectories_copy[keypoint.name]['y'] = kp_trajectories_copy[keypoint.name]['y'].tolist()
            path = os.path.join(self.save_data, 'filled_keypoints_trajectories.json')
            to_json(kp_trajectories_copy, path)
            
        return kp_trajectories

    def moving_average(self, data, window):
        ret = np.cumsum(data, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        data[window - 1:] = ret[window - 1:]  / window
        return np.array(data)

    def filter_trajectories(self, kp_trajectories, window=4, save=True):
        for keypoint in Model25_KP:
            kp_trajectories[keypoint.name]['x'] = self.moving_average(kp_trajectories[keypoint.name]['x'], window)
            kp_trajectories[keypoint.name]['y'] = self.moving_average(kp_trajectories[keypoint.name]['y'], window)
            
        kp_trajectories_copy = kp_trajectories.copy()

        if save:
            for keypoint in Model25_KP:
                kp_trajectories_copy[keypoint.name]['x'] = kp_trajectories_copy[keypoint.name]['x'].tolist()
                kp_trajectories_copy[keypoint.name]['y'] = kp_trajectories_copy[keypoint.name]['y'].tolist()
            path = os.path.join(self.save_data, 'filtered_keypoints_trajectories.json')
            to_json(kp_trajectories_copy, path)

        return kp_trajectories

    ## Angles

     # https://stackoverflow.com/questions/69154914/calculating-angles-of-body-skeleton-in-video-using-openpose
    def get_angle(self, edge1,  edge2, points, frame):
        assert tuple(sorted(edge1)) in edges
        assert tuple(sorted(edge2)) in edges
        edge1 = set(edge1)
        edge2 = set(edge2)
        mid_point = edge1.intersection(edge2).pop()
        a = (edge1-edge2).pop()
        b = (edge2-edge1).pop()

        v1 = [points[Model25_KP(mid_point).name]['x'][frame]-points[Model25_KP(a).name]['x'][frame],
            points[Model25_KP(mid_point).name]['y'][frame]-points[Model25_KP(a).name]['y'][frame]]
        v2 = [points[Model25_KP(mid_point).name]['x'][frame]-points[Model25_KP(b).name]['x'][frame],
        points[Model25_KP(mid_point).name]['y'][frame]-points[Model25_KP(b).name]['y'][frame]]

        angle = (math.degrees(np.arccos(np.dot(v1,v2)
                                        /(np.linalg.norm(v1)*np.linalg.norm(v2)))))
        return angle

    def get_oriented_angle(self, edge1,  edge2, points, frame):    
        assert tuple(sorted(edge1)) in edges
        assert tuple(sorted(edge2)) in edges
        
        v1 = [points[Model25_KP(edge1[0]).name]['x'][frame]-points[Model25_KP(edge1[1]).name]['x'][frame],
            points[Model25_KP(edge1[0]).name]['y'][frame]-points[Model25_KP(edge1[1]).name]['y'][frame]]

        v2 = [points[Model25_KP(edge2[0]).name]['x'][frame]-points[Model25_KP(edge2[1]).name]['x'][frame],
            points[Model25_KP(edge2[0]).name]['y'][frame]-points[Model25_KP(edge2[1]).name]['y'][frame]]

        angle = (math.degrees(np.arccos(np.dot(v1,v2)   
                                /(np.linalg.norm(v1)*np.linalg.norm(v2)))))
        return angle

    #TODO: Proccess angles
    def get_body_angle(self, body_part, points, frame, oriented=False):
        edge1, edge2 = get_body_edges(body_part)
        if not oriented:
            angles = self.get_angle(edge1, edge2, points, frame)
        else:
            angles = self.get_oriented_angle(edge1, edge2, points, frame)
        return angles

    def proccess_angles(self, kp_trajectories, save=True):
        kp_angles = {}

        total_frames = len(kp_trajectories[Model25_KP.LKnee.name]['x'])
        for joint in JOINT_ANGLES:
            kp_angles[joint.name] = []
            for frame in range(total_frames):
                if 'Knee' in joint.name:
                    kp_angles[joint.name].append( 180. - self.get_body_angle(joint.name, kp_trajectories, frame))
                elif 'Hip' in joint.name:
                    kp_angles[joint.name].append( 180. - self.get_body_angle(joint.name, kp_trajectories, frame, oriented=True))
                elif 'Ankle' in joint.name:
                    kp_angles[joint.name].append( self.get_body_angle(joint.name, kp_trajectories, frame, oriented=True) - 90.)

        if save:
            path = os.path.join(self.save_data, 'joint_angles.json')
            to_json(kp_angles, path)

        return kp_angles

    ## Plots    

    def plot_trajectory(self, kp_trajectory, enum_trajectory, show=True, save=False, save_format='pdf'):
        x_axis = range(len(kp_trajectory['x']))
        plt.plot(x_axis, kp_trajectory['x'], label='x')
        plt.plot(x_axis, kp_trajectory['y'], label='y')
        plt.xlabel('Frame')
        plt.ylabel('Position')
        plt.title(f'{enum_trajectory.name} spatial trajectory')
        plt.grid('on')
        plt.legend()

        if save:
            plt.savefig( os.path.join(self.save_figures, f'{enum_trajectory.name}_trajectory.{save_format}'), bbox_inches='tight', pad_inches=0.1 )

        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_pose_trajectories_matrix(self, kp_trajectories, show=True, save=False, save_format='pdf', prefix=''):
        x_axis = range(len(kp_trajectories[Model25_KP.LKnee.name]['x'])) # Generic x_axis range
        
        n_row = 5
        n_col = 5
        fig, axs = plt.subplots(n_row, n_col)
        
        nj = 0
        ni = 0
        for keypoint in Model25_KP:
            axs[ni, nj].plot(x_axis, kp_trajectories[keypoint.name]['x'], label='x')
            axs[ni, nj].plot(x_axis, kp_trajectories[keypoint.name]['y'], label='y')
            # axs[ni, nj].set_xlabel('Frame')
            # axs[ni, nj].set_ylabel('Position')
            axs[ni, nj].set_title(f'{keypoint.name}')
            axs[ni, nj].grid('on')
            # axs[ni, nj].legend()
            ni = ni + 1 if not nj+1 < n_col else ni
            nj = nj + 1 if nj+1 < n_col else 0

        # Single legend
        lines_labels = [axs[0, 0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)

        # Set general title
        fig.suptitle('Spatial Trajectory')

        plt.tight_layout()

        if save:
            plt.savefig( os.path.join(self.save_figures, f'{prefix}trajectories_matrix.{save_format}'), bbox_inches='tight', pad_inches=0.1 )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_angle_matrix(self, kp_angles, show=True, save=False, save_format='pdf', prefix=''):
        x_axis = range(len(kp_angles[Model25_KP.LKnee.name])) # Generic x_axis range
        
        n_row = 3
        n_col = 2
        fig, axs = plt.subplots(n_row, n_col)
        
        nj = 0
        ni = 0
        for keypoint in JOINT_ANGLES:
            axs[ni, nj].plot(x_axis, kp_angles[keypoint.name], RIGHT_COLOR if 'R' in keypoint.name[0] else LEFT_COLOR )
            # axs[ni, nj].set_xlabel('Frame')
            # axs[ni, nj].set_ylabel('Position')
            axs[ni, nj].set_title(f'{keypoint.name}')
            axs[ni, nj].grid('on')
            # axs[ni, nj].legend()
            ni = ni + 1 if not nj+1 < n_col else ni
            nj = nj + 1 if nj+1 < n_col else 0

        # Single legend
        lines_labels = [axs[0, 0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)

        # Set general title
        fig.suptitle('Angles')

        plt.tight_layout()

        if save:
            plt.savefig( os.path.join(self.save_figures, f'{prefix}angles_matrix.{save_format}'), bbox_inches='tight', pad_inches=0.1 )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_sample_proprocess(self, raw_data, preprocess_data, descriptions, show=True, save=True, prefix='', save_format='pdf'):
        x_axis = range(len(raw_data))
        plt.plot(x_axis, raw_data, 'r', label=descriptions['label_1'])
        plt.plot(x_axis, preprocess_data, '--b', label=descriptions['label_2'])
        plt.title(descriptions['title'])
        plt.xlabel('Frames')
        plt.ylabel('Trajectory')
        plt.grid('on')
        plt.legend()

        if save:
            save_name = descriptions['save_name']
            plt.savefig( os.path.join(self.save_figures, f'{save_name}.{save_format}'), bbox_inches='tight', pad_inches=0.1 )

        if show:
            plt.show()
        else:
            plt.close() 



opEn_subject01 = OPEngine('subject_test', 'subjects')
# trajectories_by_kp = opEn_subject01.proccess_trajectories()
raw_trajectories_by_kp = opEn_subject01.load_kp_trajectories('raw_keypoints_trajectories.json')
filled_trajectories_by_kp = opEn_subject01.load_kp_trajectories('filled_keypoints_trajectories.json')
# filled_trajectories_by_kp = opEn_subject01.handle_missing_keypoints(trajectories_by_kp)
filtered_trajectories_by_kp = opEn_subject01.filter_trajectories(filled_trajectories_by_kp)

## Example filled data Mauricio
# sample = Model25_KP.RElbow.name
# axis = 'x'
# descriptions = {
#     'label_1': 'raw data',
#     'label_2': 'filled data',
#     'title': 'Fill missing values process',
#     'save_name': 'sample_missing_values'
# }
# raw_data = raw_trajectories_by_kp[sample][axis]
# sample_data = filled_trajectories_by_kp[sample][axis]

## Example filtered data Mauricio
sample = Model25_KP.RSmallToe.name
axis = 'x'
descriptions = {
    'label_1': 'raw data',
    'label_2': 'filtered data',
    'title': 'Filter data process',
    'save_name': 'sample_filter_data'
}
raw_data = raw_trajectories_by_kp[sample][axis]
sample_data = filtered_trajectories_by_kp[sample][axis]

opEn_subject01.plot_sample_proprocess(raw_data, sample_data, descriptions, show=False)




# filtered_trajectories = opEn_subject01.load_kp_trajectories('filtered_keypoints_trajectories.json')
# angles = opEn_subject01.proccess_angles(filtered_trajectories)
# opEn_subject01.plot_angle_matrix(angles, show=False, save=True, prefix='')

# opEn_subject01.plot_trajectory(trajectories_by_kp[Model25_KP.LKnee.name], Model25_KP.LKnee, show=False, save=True)
# opEn_subject01.plot_pose_trajectories_matrix(trajectories_by_kp, show=False, save=True, prefix='raw_')
# opEn_subject01.plot_pose_trajectories_matrix(filled_trajectories, show=False, save=True, prefix='filled_')
# opEn_subject01.plot_pose_trajectories_matrix(filtered_trajectories, show=False, save=True, prefix='filled_filtered_')

