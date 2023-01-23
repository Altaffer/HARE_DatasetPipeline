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

rbg_K = np.array([[500, 0, PI_W/2],
                  [0, 500, PI_H/2],
                  [0, 0, 1]])
noir_K = np.array([[500, 0, PI_W/2],
                  [0, 500, PI_H/2],
                  [0, 0, 1]])
flir_K = np.array([[500, 0, F_W/2],
                   [0, 500, F_H/2],
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
    
    def stack(self, time):
        # check if all images are not none
        if self.rgb and self.flir and self.noir:        
            s = {"rgb": self.rgb, 
                "noir": self.noir, 
                "flir": self.flir}
            loc = os.path.join(self.dir.path, "stacked_"+str(time.secs) + '.' + str(time.nsecs)+".bin")
            # send stack to binary file
            pickle.dump(s, open(loc, "wb"))
            
            rot = R.from_quat([self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]).as_matrix()
            trans = np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z])

            # use K to compute vision on the ground
            rgb_ground_points = self.visionCone(PI_W, PI_H, rot, trans, rbg_K)
            noir_ground_points = self.visionCone(PI_W, PI_H, rot, trans, noir_K)
            flir_ground_points = self.visionCone(F_W, F_H, rot, trans, flir_K)

            # print(rgb_ground_points[0])

            # put data into reasonable format
            if POINTS_FORMAT == 1:
                # save a list of all of the ground points (ie. xma = {xma_rgb, xma_noir, xma_flir})
                xma = [rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0]]
                yma = [rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1]]
                xmi = [rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0]]
                ymi = [rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1]]

            elif POINTS_FORMAT == 2:
                # save all the data (ie. xma = {xma_com, xma_rgb, xma_noir, xma_flir})
                xma = [rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0], min(rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0])]
                yma = [rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1], min(rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1])]
                xmi = [rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0], min(rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0])]
                ymi = [rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1], min(rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1])]

            else: # POINTS_FORMAT == 0 as well
                # figure out the intersection of the ground points such that clicks will only register if seen by all 3 cameras
                xma = min(rgb_ground_points[0][0], noir_ground_points[0][0], flir_ground_points[0][0])
                yma = min(rgb_ground_points[0][1], noir_ground_points[0][1], flir_ground_points[0][1])
                xmi = min(rgb_ground_points[1][0], noir_ground_points[1][0], flir_ground_points[1][0])
                ymi = min(rgb_ground_points[1][1], noir_ground_points[1][1], flir_ground_points[1][1])


            # create 4x4 matrix for pose
            pose = np.concatenate((rot, np.expand_dims(trans, axis=0).T), axis=1)
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)


            # save stack save location, pose, date, and coords to dataframe
            self.parsed.loc[len(self.parsed.index)] = [xma, yma, xmi, ymi, pose, loc]


    def done(self):
        self.parsed.to_pickle(os.path.join(self.dir.path, "parsed.pkl"))


    def get_frames(self, x, y):
        return self.parsed.loc[(self.parsed["xmax"] > x) & (self.parsed["xmin"] < x) &
                        (self.parsed["ymax"] > y) & (self.parsed["ymin"] < y)]['save_loc']

    def visionCone(self, w, h, r, t, K=False):
        # teh corners of the image
        ul = np.array([0,0,1])
        # ur = np.array([0,w-1,1])
        br = np.array([h-1,w-1,1])
        # bl = np.array([h-1,0,1])
        corners = [ul, br]


        vizCone = []
        for corner in corners:
            # project corner to meters
            if K is not False:
                tmp = np.linalg.inv(K)@corner
            else:
                tmp = corner
            # scale corner to altitude
            tmp = tmp * ASSUMED_ALTITUDE
            # rotate and translate corner by pose
            # print(r, t, tmp)
            tmp = np.linalg.inv(r)@(tmp - t) # this may be backwards
            vizCone.append(tmp[:2])

        return vizCone

@dataclass
class Dir:
    path: str
    date: str
    bag: str = None
    clicks: str = None

