import matplotlib.pyplot as plt
from model25_keypoints import Model25_KP

LEFT_COLOR = 'r' # Color plot by standard
RIGHT_COLOR = 'b' # Color plot by standard
plt.rcParams["figure.figsize"] = (12, 9) # Size plot by standard
plt.rcParams['figure.dpi'] = 140 # output dpi for png images

# Posible edges in model_25 openpose
edges = {(0, 1), (0, 15), (0, 16), (1, 2), (1, 5), 
(1, 8), (2, 3), (3, 4), (5, 6), (6, 7), 
(8, 9), (8, 12), (9, 10), (10, 11), (11, 22), 
(11, 24), (12, 13), (13, 14), (14, 19), (14, 21), 
(15, 17), (16, 18), (19, 20), (22, 23)}

JOINT_ANGLES = [Model25_KP.LKnee, Model25_KP.RKnee, 
            Model25_KP.LHip, Model25_KP.RHip,
            Model25_KP.LAnkle, Model25_KP.RAnkle
            ]