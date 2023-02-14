import pandas as pd
from banners import *
import csv_relations
import numpy as np
import cv2

cols = ['stack_location', 'robot_x', 'robot_y', 'pixel_points', 'health']

PI_W = 1920
PI_H = 1080
PI_FOV = 55
F_W = 640
F_H = 512
F_FOV = 55

rbg_K = np.array([[679.0616691224743, 0.0, 866.4845535612815],
                  [0.0, 679.7611413287517, 593.4758325974849],
                  [0, 0, 1]])
rbg_dist = np.array([-0.22658516071521045, 0.04799750990896024, 0.0003471450894650341, -0.00048311114989200163])

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
                print(pts)
                pxs = self.fProject(pts, rbg_K)
                c2i_df.loc[len(c2i_df.index)] = [location, p[0,3], p[1,3], pxs, health]
        return c2i_df


    # make points relative to the robot pose 
    def convert_to_local(self, pose, pts):
        p = np.asarray(pts.loc[:,['x', 'y']])
        p = p - np.array([pose[0,3], pose[1,3]])
        p = np.concatenate((p, np.array([[10]])), axis=1)
        # print(p.shape)
        return np.linalg.inv(np.array(pose[:3,:3]))@p.T

    # undistort the raw frame for input to pretty_image
    def undistort(self, img, mtx, dist, w, h):
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst
        
    # forward project points into image
    def fProject(self, points, K):
        # add an altitude to points
        # h = np.full((1,points.shape[0]), 10)
        # points = np.concatenate((points, h), axis=1)
        # print("raw points from input", points)
        p = K@points
        p = p / p[-1, :]
        return p[:2, :]
        