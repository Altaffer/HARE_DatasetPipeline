import cv2
import rosbag
import utm
import os 
import rospy
from dataclasses import dataclass
import pandas as pd
import pickle
import csv

import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import UInt8
from sensor_msgs.msg import Imu, NavSatFix, Image
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Pose
from scipy.spatial.transform import Rotation as R

ASSUMED_ALTITUDE = 10 #m

POINTS_FORMAT = 0

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

# TODO: rgb extrinsics 


noir_K = np.array([[677.6841664243713, 0.0, 927.0775869012278],
                  [0.0, 678.3384456258163, 545.5178145289105],
                  [0, 0, 1]])
flir_K = np.array([[250, 0, F_W/2],
                   [0, 250, F_H/2],
                   [0, 0, 1]])


@dataclass
class Dir:
    path: str
    date: str
    bag: str = None
    clicks: str = None

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
            loc = str(os.path.join(os.path.join(self.dir.path, 'stacks/'), "stacked_"+str(time.secs) + '.' + str(time.nsecs)+".bin"))
            # send stack to binary file
            with open(loc, 'wb') as f:
                # print('dumping stack')
                pickle.dump(s, f)
            

            rot = R.from_quat([self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]).as_matrix()
            trans = np.array([[self.pose.position.x, self.pose.position.y, self.pose.position.z]]).T

            # make a 4x4 pose matrix
            pose = np.concatenate((rot, trans), axis=1)
            pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)

            # save stack save location, pose, date, and coords to dataframe
            self.parsed.loc[len(self.parsed.index)] = [pose, loc]
            self.flir = None
            self.noir = None
            self.rgb = None

    
    # undistort the raw frame for input to pretty_image
    def undistort(self, img, mtx, dist):
        w = img.shape[1]
        h = img.shape[0]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def done(self):  
        self.parsed.to_pickle(os.path.join(self.dir.path, "parsed.pkl"))

    
    def visCone(self, pose, FOV_angle, eps=0):        
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


aug_cols = ["x", "y", "health"]

class CSVAugmented:
    def __init__(self, file, root):
        self.csv_file = file
        self.root = root
        self.click_data = pd.DataFrame(columns=aug_cols)

    def done(self):  
        self.click_data.to_pickle(os.path.join(self.root, "related.pkl"))
        # with open(os.path.join('data/test_dir/', "related.pkl"), 'wb') as f:
        #     pickle.dump(self.__dict__, f, protocol=2)


    # def uptake(self):
    #     self.click_data = pd.read_pickle(os.path.join(self.root, "related.pkl"))


    def csv_read(self):
        origin = None
        with open(self.csv_file) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                if origin == None:
                    origin = [float(line[0]), float(line[1]), float(line[2])]
                # breakdown line
                u = utm.from_latlon(float(line[0]), float(line[1]))
                # u = pm.geodetic2ned(float(line[0]), float(line[1]), float(line[2]), origin[0], origin[1], origin[2])
                # print(u)
                health = int(line[-1])
                
                self.click_data.loc[len(self.click_data.index)] = [u[0], u[1], health]

    """ get the clicks within the FOV """
    def get_pts(self, pose, cone):
        got_points = self.click_data.loc[(self.click_data['x'] > cone[0] + pose[0,3] ) & 
                            (self.click_data['x'] < cone[1] + pose[0,3] ) & 
                            (self.click_data['y'] > cone[2] + pose[1,3] ) & 
                            (self.click_data['y'] < cone[3] + pose[1,3] )]
        return got_points
        

# def done(fname, obj):
#     with open(fname, 'wb') as f:
#         pickle.dump(obj, f, protocol=2)