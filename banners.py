#!/usr/bin/env python3

import cv2
import rosbag
import argparse
import utm
import yaml
import cv2
import pathlib 
import rospy

import numpy as np

from pathlib import Path
from cv_bridge import CvBridge
from std_msgs.msg import UInt8
from sensor_msgs.msg import Imu, NavSatFix, Image
from geometry_msgs.msg import PointStamped
from scipy.spatial.transform import Rotation as R

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
TOPICS = ['/dji_osdk_ros/gps_health', '/dji_osdk_ros/gps_position', '/dji_osdk_ros/imu', '/dji_osdk_ros/local_position', '/cam0/camera/image_raw', '/cam1/camera/image_raw', '/therm/image_raw_throttle']
CAM_LIST = ['rgb', 'noIR', 'thermal']

class CameraCombo:
    
    def __init__(self, topics, cam0_yaml, cam1_yaml, therm_yaml):
        self.cam0 = None
        self.cam0_params = self.parse_yaml(cam0_yaml)
        self.cam1 = None
        self.cam1_params = self.parse_yaml(cam1_yaml)
        self.therm = None
        self.therm_params = self.parse_yaml(therm_yaml)

        self.pos = None
        self.gps_old = None
        self.gps_new = None
        self.gps_health = None

        self.r = None
        self.t = None

        self.odom_sub = rospy.Subscriber(topics[1], NavSatFix, self.gps_pos_callback)
        self.gps_health_sub = rospy.Subscriber(topics[0], UInt8, self.gps_health_callback)
        self.cam0_sub = rospy.Subscriber(topics[4], Image, self.cam0_callback)
        self.cam1_sub = rospy.Subscriber(topics[5], Image, self.cam1_callback)
        self.therm_sub = rospy.Subscriber(topics[6], Image, self.therm_callback)

        # TODO: decide on dataset structure and formats to implement the saving protocol
        # for t in topics:
        #     Path("imgs_f/" + t).mkdir(parents=True, exist_ok=True)


    def gps_health_callback(self, msg):
        self.gps_health = msg  # update gps health
        pass


    def odom_callback(self, msg):        
        if self.gps_health is not None and self.gps_health >= 3:
            self.gps_old = self.gps_new  # if gps is healthy, take a reading
            self.gps_new = msg
        else:  # else clear stale readings
            self.gps_old = None
            self.gps_new = None  # this means that I need 2 odom_msgs before I can run the fov check/annotator 
        pass


    def cam0_callback(self, msg):
        self.cam0 = msg
        pass


    def cam1_callback(self, msg):
        self.cam1 = msg
        pass


    def therm_callback(self, msg):        
        self.therm = msg
        pass


    def parse_yaml(self, yaml_in):
        """ Parse ROS camera_calibration-formatted yaml to cv2.undistort() inputs. """

        with open(yaml_in) as f:
            ci = yaml.safe_load(f)

        w = ci["image_width"]
        h = ci["image_height"]
        try:
            mtx = ci["camera_matrix"]["data"].reshape((3,3))
        except KeyError:  # may be incorrect error to catch
            mtx=False
        dst = ci["distortion_coefficients"]["data"]

        return w, h, mtx, dst

    
    def odomDecode(self):
        """ decode quaternion into rotation matrix and compute aggregate affine transform since last frame 
        
        R is decoded quaternion
        T is computed from delta between self.gps_pos data, converted to (m,m,m) from (lat,lon,m) and visionCone output
        """
        gps_old = (self.gps_old['latitude'], self.gps_old['longitude'], self.gps_old['altitude'])
        gps_new = (self.gps_new['latitude'], self.gps_new['longitude'], self.gps_new['altitude'])
        head = self.pos['heading']

        r = R.from_quat(head).as_matrix()  # from_quat() assumes scalar first, head may be scalar last. watch for inaccurate behavior

        return r, gps_new - gps_old


    def visionCone(self, w, h, r, t, K=False):
        """ find corners of polyhedral vision cone in real-world units """

        ul = np.array([0,0,1])
        ur = np.array([0,w-1,1])
        br = np.array([h-1,w-1,1])
        bl = np.array([h-1,0,1])

        corners = [ul, ur, br, bl]
        vizCone = []

        for corner in corners:
            if K is not False:
                tmp = np.linalg.inv(K)@corner
            else:  # assume prerectified, no need for multiplication of corner with K^(-1)
                tmp = corner
            tmp = np.linalg.inv(r)@(tmp - t)  # variable t may need to come to meters from UTM
            # scale = self.gps_pos['altitude']/tmp[-1]  # need a scale factor to account for altitude/depth, wanna use the laser, not GPS
            # tmp *= scale
            vizCone.append(tmp)

        return vizCone


    def FOV(self):
        """ construct FOV

        Back projection to see if clicks are within FOV of a given frame 
        
        Given field annotations, drone GPS+IMU odometry, camera geometry
        Return cone of vision in NED/GPS coordinates

        ASSUMES THE GROUND IS FLAT WITHIN THE FOV - load DEM/other map for better representation of the ground
        --> fovCheck probably gets significantly more expensive without flatness... Does it?
            --> is cost increase worth it? 
                + position of pins will not vary much within O(0.25m) approx <-> true altitude variance

        """
        tmp = []
        frame = (self.cam0, self.cam1, self.therm)
        params = (self.cam0_params, self.cam1_params, self.therm_params) 
        if frame != (None, None, None) and self.gps_health >= 3:
            self.r, self.t = self.odomDecode()
            for i, fr in enumerate(frame):  # rgb (30Hz), noIR (30Hz), and FLIR (5Hz) frames present
                w,h,mtx,_ = params[i]
                vizCone = self.visionCone(w, h, self.r, self.t, mtx)
                tmp.append((fr, params[i], vizCone))
        else:
            tmp.append((fr, params[i], None))

        self.cam0 = None
        self.cam1 = None
        self.therm = None

        return tmp


    def fovCheck(self, visConeEntry, annotations):
        """ taget in FOV check

        given self.FOV != None output and annotation set, see if annotations are in the FOV

        """
        inView = None
        _, _, cone = visConeEntry
        if cone is not None:  # corners = [ul, ur, br, bl]
            inView = []
            normals = []
            for i in range(len(cone)):
                try:
                    a,b,_ = cone[i]
                    c,d,_ = cone[i+1]
                except IndexError:
                    a,b,_ = cone[i]
                    c,d,_ = cone[0]
                normals.append(np.array([d-b, a-c]))

            for i, annotation in enumerate(annotations):
                checks = []
                for normal in normals:
                    checks.append(annotation@normal)
                if np.all(checks < 0):  # annotation is in the cone
                    inView.append(i)

        return inView


    def annotator(self, inView, annotations):
        """ construct annotation string to append to filename using inView """
        tmp = ""

        for i in inView:
            pin = annotations[i]
            
             
            tmp += str(pin)
            tmp += "-"
            
        return tmp


def main(bagname, _yamls, _csv):
    """ run it all """
    annotations = np.genfromtxt(_csv, delimiter=',')

    rospy.init_node('data_tube')
    cc = CameraCombo(TOPICS, _yamls)

    while True:
        if cc.cam0 == None and cc.cam1 == None and cc.therm == None: 
            visCone = None
        elif cc.pos == None:
            visCone = None
        else:
            visCone = cc.FOV()

        for i, entry in enumerate(visCone):
            if entry[-1] is not None:
                inView = cc.fovCheck(entry, annotations)
                if inView is not None:
                    inViewStr = cc.annotator(inView, entry, annotations)
                else: 
                    inViewStr = "NoTargetsInView"
            else:
                print("Problem constructing field of view.")
            # savename = CAM_LIST[i] + inViewStr  # need to know exactly what comes in 
            # with open(savename, 'xw'):
            #    

    return 0


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Stitch images and position data to GPS locations of targets.')
    parser.add_argument('bagname', type=str)
    parser.add_argument('yaml', type=str, required=False)
    parser.add_argument('csv', type=str, required=False)
    args =parser.parse_args()
    
    main(args.bagname, args.yaml)
