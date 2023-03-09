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
import sqlite3
import io

from cv_bridge import CvBridge
from std_msgs.msg import UInt8
from sensor_msgs.msg import Imu, NavSatFix, Image
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Pose
from scipy.spatial.transform import Rotation as R

bridge = CvBridge()


rgb_K = np.array([[3328.72744368, 0.0, 985.2442405],
                  [0.0, 3318.46036526, 489.0953335],
                  [0, 0, 1]])
rgb_dist = np.array([-0.33986049, -0.49477998,  0.00326809, -0.00230553])



@dataclass
class Dir:
    path: str
    date: str
    bag: str = None
    clicks: str = None

class DBConnector:
    def __init__(self) -> None:
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convert_array)
        # con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
        self.db_c = sqlite3.connect('drone_disease.db', detect_types=sqlite3.PARSE_DECLTYPES)

    def adapt_array(self, arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

    def setupArrays(self):
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", self.convert_array)
        con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)

    def diagnostic(self, table, max=0):
        cur = self.db_c.cursor()
        limit = ';'
        if max != 0:
            limit = f" LIMIT {max};"
        res = cur.execute(f"SELECT * FROM {table}" + limit)
        print(res.fetchall())

    def getFrom(self, what, where, max=0):
        cur = self.db_c.cursor()
        limit = ';'
        if max != 0:
            limit = f" LIMIT {max};"
        res = cur.execute(f"SELECT {what} FROM {where}" + limit)
        return res.fetchall()

        
    

    
    




class Banners:
    def __init__(self, flight_name='night_out'):
        self.dir = None
        self.rgb = None
        self.noir = None
        self.flir = None
        self.pose = None # this is just pose, remove all the stamping and covariance 
        self.flight_name = flight_name
        self.is_new = False
        # cols = ["pose", "save_loc"]
        self.db_c = sqlite3.connect('drone_disease.db')
        cur = self.db_c.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='flight_{flight_name}';")
        if len(res.fetchall()) == 0:
            cur.execute(f"CREATE TABLE flight_{flight_name}((x REAL, y REAL, z REAL, q REAL, u REAL, a REAL, t REAL, r_save_loc TEXT, n_save_loc TEXT, f_save_loc TEXT))")
            cur.execute(f"CREATE TABLE clicks_{flight_name}(x REAL, y REAL, health INTEGER)")
            self.db_c.commit()
            self.is_new = True
        self.rg_fov_check = []
        self.no_fov_check = []
        self.fl_fov_check = []

    def convertAndSave(self, data, sensor, time):
        vcimg = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        location = os.path.join(self.dir.path, 'stacks/')
        location = os.path.join(location, sensor)
        location = os.path.join(location, "img_"+str(time.secs) + '.' + str(time.nsecs)+".png")
        cv2.imwrite(location, vcimg)
        return location


    def stack(self, time):
        # check if all images are not none
        if self.rgb and self.flir and self.noir:
            # save each image file
            rgb_location = self.convertAndSave(self.rgb, 'rgb', time)
            noir_location = self.convertAndSave(self.noir, 'noir', time)
            flir_location = self.convertAndSave(self.flir, 'flir', time)            

            cur = self.db_c.cursor()
            cur.execute(f"INSERT INTO flight_{self.flight_name} (x, y, z, q, u, a, t, r_save_loc, n_save_loc, f_save_loc) VALUES ({self.pose.position.x}, {self.pose.position.y}, {self.pose.position.z}, {self.pose.orientation.x}, {self.pose.orientation.y}, {self.pose.orientation.z}, {self.pose.orientation.w}, '{rgb_location}', '{noir_location}', '{flir_location}')")
            self.db_c.commit()

            self.flir = None
            self.noir = None
            self.rgb = None

    def csv_read(self, csv_file):
        data = []
        with open(csv_file) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # breakdown line
                u = utm.from_latlon(float(line[0]), float(line[1]))
                health = int(line[-1])
                
                data.append((u[0], u[1], health))
        cur = self.db_c.cursor()
        cur.executemany(f"INSERT INTO clicks_{self.flight_name} VALUES (?, ?, ?)", data)
        self.db_c.commit()
        cur.execute(f"SELECT * FROM clicks_{self.flight_name}")
        res = cur.fetchall()
        for r in res:
            print(r)




class Relater:
    def __init__(self, banners_id, flight_name='get_lucky'):
        self.db_c = sqlite3.connect('drone_disease.db')
        self.ban_id = banners_id
        self.flight_name = flight_name
        cur = self.db_c.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='relations_{flight_name}';")
        if len(res.fetchall()) == 0:
            cur.execute(f"CREATE TABLE relations_{flight_name} (img_loc TEXT, pred_pixel_x array, pred_pixel_y array, blob_center_x array, blob_center_y array)")
            self.db_c.commit()

    def makePoseMatrix(self, trans, rot):
        rot = R.from_quat(rot).as_matrix()
        trans = np.array([trans]).T

        # make a 4x4 pose matrix
        pose = np.concatenate((rot, trans), axis=1)
        pose = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)
        return pose

    def visCone(self, pose, FOV_angle, eps=0):        
        FOV_angle -= eps    # subtract fudge factor, defaulted to 0
        FOV_angle *= np.pi  # convert to radians
        FOV_angle /= 180    
        FOV_angle /= 2      # cone edges relative to center

        horz = pose[2,3] * np.sin(FOV_angle)
        vert = pose[2,3] * np.sin(FOV_angle)
        cone = [-horz, horz, -vert, vert]
        return cone

    def getClicksInFOV(self, pos, fov):
        cone = self.visCone(pos, fov)
        q = f"SELECT x, y FROM clicks_{self.ban_id} WHERE (x BETWEEN {cone[0] + pos[0,3]} and {cone[1] + pos[0,3]}) and (y BETWEEN {cone[2] + pos[1,3]} and {cone[3] + pos[1,3]});"
        return q

    def clicks_to_image(self, FOV_angle):
        # iterate over the images and poses
        cur = self.db_c.cursor()
        res = cur.execute(f"SELECT x, y, z, q, u, a, t, r_save_loc FROM flight_{self.ban_id};")
        predictions = []
        for r in res.fetchall():
            # print(r)
            pose = self.makePoseMatrix([float(r[0]), float(r[1]), float(r[2])], [float(r[3]), float(r[4]), float(r[5]), float(r[6])])
            query = self.getClicksInFOV(pose, FOV_angle) # returns the location of the clicks that can be seen
            pts = cur.execute(query)
            pts = pts.fetchall()
            print("points", pts)
            if len(pts) > 0:
                img_pts_x = []
                img_pts_y = []
                for point in pts:
                    point = self.convert_to_local(pose, point)
                    # print(pts)
                    pxs = self.fProject(point, rgb_K)
                    print(pxs[1,:])
                    img_pts_x.append(pxs[0,:][0])
                    img_pts_y.append(pxs[1,:][0])
                predictions.append((r[7], np.array(img_pts_x), np.array(img_pts_y)))
                # print(q)
        # print(predictions)
        cur.executemany("INSERT INTO relations_get_lucky (img_loc, pred_pixel_x, pred_pixel_y) VALUES (?, ?, ?)", predictions)
        self.db_c.commit()
          

    def fProject(self, points, K):
        p = K@points
        p = p / p[-1, :]
        return p[:2, :]

    
    def convert_to_local(self, pose, pts):
        # make points 4d
        p = np.asarray([[pts[0], pts[1]]])
        p = np.concatenate((p, np.zeros((1, p.shape[0]))), axis=1)
        p = np.concatenate((p, np.ones((1, p.shape[0]))), axis=1)
        # print(p)
        tf_pts = np.linalg.inv(pose)[:3, :]@p.T
        # print(p.shape)
        return tf_pts