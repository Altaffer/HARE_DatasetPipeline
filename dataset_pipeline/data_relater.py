import pandas as pd
from banners import *
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R

bridge = CvBridge()

cols = ['stack_location', 'robot_x', 'robot_y', 'pixel_points', 'health']

PI_W = 1920
PI_H = 1080
PI_FOV = 55
F_W = 640
F_H = 512
F_FOV = 55

rgb_K = np.array([[3328.72744368, 0.0, 985.2442405],
                  [0.0, 3318.46036526, 489.0953335],
                  [0, 0, 1]])
rgb_dist = np.array([-0.33986049, -0.49477998,  0.00326809, -0.00230553])

noir_K = np.array([[677.6841664243713, 0.0, 927.0775869012278],
                  [0.0, 678.3384456258163, 545.5178145289105],
                  [0, 0, 1]])
noir_dist = np.array([-0.22664276217328275, 0.047805254694810353, -0.0008087741168388509, -6.089873727806487e-05])

flir_K = np.array([[250, 0, F_W/2],
                   [0, 250, F_H/2],
                   [0, 0, 1]])

class Relater:
    def __init__(self):
       self.cc = CameraCombo()
       
    def clicks_to_image(self, clicks_o, FOV_angle):
        c2i_df = pd.DataFrame(columns=cols)
        for idx, row in self.cc.parsed.iterrows():
            p = row['pose']
            pts = clicks_o.get_pts(p, self.cc.visCone(p, FOV_angle))
            if len(pts.index):
                health = pts['health']
                location = row['save_loc']
                pts = self.convert_to_local(p, pts)
                # print(pts)
                pxs = self.fProject(pts, rgb_K)
                c2i_df.loc[len(c2i_df.index)] = [location, p[0,3], p[1,3], pxs, health]
        return c2i_df


    # make points relative to the robot pose 
    def convert_to_local(self, pose, pts):
        # make points 4d
        p = np.asarray(pts.loc[:,['x', 'y']])
        p = np.concatenate((p, np.zeros((1, p.shape[0]))), axis=1)
        p = np.concatenate((p, np.ones((1, p.shape[0]))), axis=1)
        # print(p)
        tf_pts = np.linalg.inv(pose)[:3, :]@p.T
        # print(p.shape)
        return tf_pts
    
    def convert_to_local_og(self, pose, pts):
        # pitch down 90deg
        offset = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        p = np.asarray(pts.loc[:,['x', 'y']])
        p = p - np.array([pose[0,3], pose[1,3]])
        p = np.concatenate((p, np.array([[pose[2,3]]])), axis=1)
        # print(p.shape)
        # return np.linalg.inv(np.array(pose[:3,:3])@offset)@p.T
        rot = R.from_matrix(np.array(pose[:3,:3]))
        eul = rot.as_euler('xyz', degrees=True)
        e = R.from_euler('xyz', [eul[0]*-1, eul[1]*-1, eul[2]], degrees=True)
        rot2 = e.as_matrix()
        return offset@p.T
        
    # forward project points into image
    def fProject(self, points, K):
        # add an altitude to points
        # h = np.full((1,points.shape[0]), 10)
        # points = np.concatenate((points, h), axis=1)
        # print("raw points from input", points)
        p = K@points
        p = p / p[-1, :]
        return p[:2, :]

    # def pretty_image(df, idx, D):
    #     row = df.iloc[idx]
    #     with open(row['stack_location'], 'rb') as f:
    #         stack = pickle.load(f)
            
    #         rgb = bridge.imgmsg_to_cv2(stack['rgb'], desired_encoding='passthrough')
    #         print(rgb.shape)
    #         rgb = D.undistort(rgb, rgb_K, rgb_dist)
    #         points = row['pixel_points']
    #         print(points[0].item(), points[1])
    #         rgb = cv2.circle(rgb, (int(points[0].item()), int(points[1].item())), 25, (0, 0, 255), 4)
    #         # plt.imshow(rgb)
    #         return rgb
        