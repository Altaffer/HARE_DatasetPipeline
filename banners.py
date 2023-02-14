#!/usr/bin/env python3

import cv2
import rosbag
import utm
import os 
import rospy
from dataclasses import dataclass
import pandas as pd
import pickle

import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import UInt8
from sensor_msgs.msg import Imu, NavSatFix, Image
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Pose
from scipy.spatial.transform import Rotation as R

import csv_relations

ASSUMED_ALTITUDE = 10 #m

POINTS_FORMAT = 0

PI_W = 1920
PI_H = 1080
PI_FOV = 55
F_W = 640
F_H = 512
F_FOV = 55

rbg_K = np.array([[679.0616691224743, 0.0, 866.4845535612815],
                  [0.0, 679.7611413287517, 593.4758325974849],
                  [0, 0, 1]])

offset_rgb = np.array([[1, 0, 0.48],
                       [0, -1, -0.3],
                       [0, 0, 1]])
noir_K = np.array([[677.6841664243713, 0.0, 927.0775869012278],
                  [0.0, 678.3384456258163, 545.5178145289105],
                  [0, 0, 1]])
flir_K = np.array([[250, 0, F_W/2],
                   [0, 250, F_H/2],
                   [0, 0, 1]])

"""
## The role of this software is to 
      1.) compute the intersection of the cone of vision with the ground
            i. I don't think I need to load images
      2.) check that region for GPS clicks
      3.) mark each frame with visible GPS clicks with the regions designated by the GPS clicks
            i. I will need to copy video frames to directories dedicated for annotations
            ii. What is the format of an annotation?
"""

### idea: take topics and msg types as lists w/ 1:1 correspondence, zip lists
# TOPICS = ['/current_pose', '/cam0/camera/image_raw', '/cam1/camera/image_raw', '/therm/image_raw_throttle']
CAM_LIST = ['rgb', 'noIR', 'thermal']

cols = ["pose", "save_loc"]

class CameraCombo:
    def __init__(self):
        self.dir = None
        self.rgb = None
        self.noir = None
        self.flir = None
        self.pose = None # this is just pose, remove all the stamping and covariance 
        self.parsed = pd.DataFrame(columns=cols)
        self.rg_fov_check = []
        self.no_fov_check = []
        self.fl_fov_check = []
    
    def stack(self, time):
        # check if all images are not none
        if self.rgb and self.flir and self.noir:        
            s = {"rgb": self.rgb, 
                "noir": self.noir, 
                "flir": self.flir}
            loc = os.path.join(os.path.join(self.dir.path, 'stacks/'), "stacked_"+str(time.secs) + '.' + str(time.nsecs)+".bin")
            # send stack to binary file
            with open(loc, 'wb') as f:
                pickle.dump(s, f)
            
            rot = R.from_quat([self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]).as_matrix()
            trans = np.array([[self.pose.position.x, self.pose.position.y, self.pose.position.z]]).T

            # make a 4x4 pose matrix
            pose = np.concatenate((rot, trans), axis=1)
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)

            # save stack save location, pose, date, and coords to dataframe
            self.parsed.loc[len(self.parsed.index)] = [pose, loc]


    def done(self):  
        self.parsed.to_pickle(os.path.join(self.dir.path, "parsed.pkl"))


    def uptake(self):
        self.parsed = pd.read_pickle(os.path.join(self.dir.path, "parsed.pkl"))

    
    def visCone(self, pose, FOV_angle, eps=0):
        
        ## TODO: this needs a test function/harness, i.e. put it into the frame
        
        FOV_angle -= eps    # subtract fudge factor, defaulted to 0
        FOV_angle *= np.pi  # convert to radians
        FOV_angle /= 180    
        FOV_angle /= 2      # cone edges relative to center

        horz = pose[2,3] * np.sin(FOV_angle)
        vert = pose[2,3] * np.sin(FOV_angle)
        cone = [-horz, horz, -vert, vert]
        return cone


    def pretty_image(self, pt, img_loc):
        imgs = pickle.loads(img_loc)
        row = self.parsed.loc[self.parsed['save_loc'] == img_loc]
        # subtract xmax and ymax from pt then use K to project back
        x = row.xmax - pt[0]
        y = row.ymax - pt[1]
        print(x, y)


@dataclass
class Dir:
    path: str
    date: str
    bag: str = None
    clicks: str = None

