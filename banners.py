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
F_W = 640
F_H = 512

rbg_K = np.array([[679.0616691224743, 0.0, 866.4845535612815],
                  [0.0, 679.7611413287517, 593.4758325974849],
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


## As of 2022-10-10, the bag has structure:
  /cam0/camera/image_raw                      # imagery
  /cam1/camera/image_raw                      # imagery
  /clicked_point
  /clock
  /dji_osdk_ros/attitude                      
  /dji_osdk_ros/battery_state
  /dji_osdk_ros/flight_status
  /dji_osdk_ros/gps_position                  # this one for world position
  /dji_osdk_ros/height_above_takeoff          
  /dji_osdk_ros/imu                           # this one for orientation (heading)
  /dji_osdk_ros/local_position                # this one for live altitude, camera origin/position
  /dji_osdk_ros/rc
  /dji_osdk_ros/rc_connection_status
  /dji_osdk_ros/velocity
  /initialpose
  /move_base_simple/goal
  /rosout
  /rosout_agg
  /tf
  /tf_static
  /therm/image_raw_throttle                   # imagery
  /ublox/fix
"""

### idea: take topics and msg types as lists w/ 1:1 correspondence, zip lists
# TOPICS = ['/current_pose', '/cam0/camera/image_raw', '/cam1/camera/image_raw', '/therm/image_raw_throttle']
CAM_LIST = ['rgb', 'noIR', 'thermal']

cols = ["xmax", "ymax", "xmin", "ymin", "pose", "save_loc"]

class CameraCombo:
    def __init__(self, dir_data):
        self.dir = dir_data
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
            pickle.dump(s, open(loc, "wb"))
            
            rot = R.from_quat([self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]).as_matrix()
            trans = np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z])

            # use K to compute vision on the ground
            rgb_ground_points = self.visionCone(PI_W, PI_H, rot, trans, rbg_K)
            # noir_ground_points = self.visionCone(PI_W, PI_H, rot, trans, noir_K)
            # flir_ground_points = self.visionCone(F_W, F_H, rot, trans, flir_K)

            # print(rgb_ground_points[0])

            # put data into reasonable format
            # if POINTS_FORMAT == 1:
            #     # save a list of all of the ground points (ie. xma = {xma_rgb, xma_noir, xma_flir})
            #     xma = [rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0]]
            #     yma = [rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1]]
            #     xmi = [rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0]]
            #     ymi = [rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1]]

            # elif POINTS_FORMAT == 2:
            #     # save all the data (ie. xma = {xma_com, xma_rgb, xma_noir, xma_flir})
            #     xma = [rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0], min(rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0])]
            #     yma = [rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1], min(rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1])]
            #     xmi = [rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0], min(rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0])]
            #     ymi = [rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1], min(rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1])]

            # else: # POINTS_FORMAT == 0 as well
            #     # figure out the intersection of the ground points such that clicks will only register if seen by all 3 cameras
            #     xma = min(rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0])
            #     yma = min(rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1])
            #     xmi = min(rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0])
            #     ymi = min(rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1])

            xma = rgb_ground_points[0,0]
            yma = rgb_ground_points[0,1]
            xmi = rgb_ground_points[1,0]
            ymi = rgb_ground_points[1,1]

            # create 4x4 matrix for pose
            pose = np.concatenate((rot, np.expand_dims(trans, axis=0).T), axis=1)
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)


            # save stack save location, pose, date, and coords to dataframe
            self.parsed.loc[len(self.parsed.index)] = [xma, yma, xmi, ymi, pose, loc]


    def done(self):
        # # check that all fovs are good
        # if self.rg_fov_check.count(self.rg_fov_check[0].all()) == len(self.rg_fov_check):
        #     print("all rgb FOVs are the same")
        # else:
        #     print("SOMETHING WRONG WITH RGB FOV")
        # if self.no_fov_check.count(self.no_fov_check[0].all()) == len(self.no_fov_check):
        #     print("all noir FOVs are the same")
        # else:
        #     print("SOMETHING WRONG WITH NOIR FOV")
        # if self.fl_fov_check.count(self.fl_fov_check[0].all()) == len(self.fl_fov_check):
        #     print("all flir FOVs are the same")
        # else:
        #     print("SOMETHING WRONG WITH FLIR FOV")
        
        
        self.parsed.to_pickle(os.path.join(self.dir.path, "parsed.pkl"))


    def get_frames(self, x, y):
        got_frames = self.parsed.loc[(self.parsed["xmax"] > x) & (self.parsed["xmin"] < x) &
                        (self.parsed["ymax"] > y) & (self.parsed["ymin"] < y)]['save_loc']

        print(got_frames)
        return got_frames


    def visionCone(self, w, h, r, t, K=False):
        """ Computes the world positions of frame corners.
        
        Inputs:
            w, scalar = frame width, pixels
            h, scalar = frame height, pixels
            r, matrix = rotation matrix from last frame POV, decoded from heading quaternion
            TODO: r may need to be augmented by the last (few?) headings, for rotation relative to last frame
            t, vector = translation from last frame POV 
            K, matrix = intrinsics matrix, or False if no calibration is available/needed
            
        Returns:
            vizCone, matrix = matrix whose columns are the real-world coordinates of the frame corners in
                    the camera reference frame, ordered [ ul | ur | bl | br ]
            """

        # the corners of the image
        ul = np.array([0,0,1]).T
        ur = [0,w-1,1]
        br = np.array([h-1,w-1,1]).T
        bl = [h-1,0,1]
        # corners = np.array([ul, br]).T  # corners is a 3x2 matrix
        corners = np.array([ul, ur, bl, br]).T  # corners is a 3x4 matrix

        # convert pixels to world units
        if K is not False:
            # vizCone = np.linalg.inv(K)@corners
            ul = np.dot(np.linalg.inv(K), ul)
            br = np.dot(np.linalg.inv(K), br)
        else:
            vizCone = corners
        # print('before rescale', vizCone)
        # vizCone /= vizCone[2,:]
        # vizCone *= ASSUMED_ALTITUDE  # scale to altitude
        # print('after rescale', vizCone)

        print(ul, br)
        # t_m = np.tile(t, (4, 1))
        # print(t_m.T)
        
        # complete back-projection
        # vizCone -= t_m.T  # replicate so that t is subtracted from each corner
        # vizCone = np.linalg.inv(r)@vizCone

        # create pose mtx
        pose = np.concatenate((r, np.expand_dims(t, axis=1)), axis=1)
        pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)

        # pad = np.array([[1, 1, 1, 1]])
        # h_viz = np.concatenate((vizCone, pad), axis=0)
        # vizCone = np.linalg.inv(pose)@h_viz

        ul = ul + t
        br = br + t

        yaw = np.array([[0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
        r = r@yaw

        ul = np.dot(np.linalg.inv(r), ul)
        br = np.dot(np.linalg.inv(r), br)

        print(ul, br)

        return np.array([ul, br])


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

