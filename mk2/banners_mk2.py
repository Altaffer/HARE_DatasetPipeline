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
import matplotlib.pyplot as plt
import time


from cv_bridge import CvBridge
from std_msgs.msg import UInt8
from sensor_msgs.msg import Imu, NavSatFix, Image
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Pose
from scipy.spatial.transform import Rotation as R

bridge = CvBridge()

BLUR_ITER = 8
BLUR_KERNEL = (15,15)
BLUR_GAUSSVAR = 10  # Very important hyperparameter
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 3
# Line thickness of 2 px
thickness = 5


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

    def getFrom(self, what, where, max=0, cond=None):
        cur = self.db_c.cursor()
        limit = ';'
        if max != 0:
            limit = f" LIMIT {max};"
        if cond == None:
            cond = ' '
        res = cur.execute(f"SELECT {what} FROM {where}" + limit + " " + cond)
        return res.fetchall()



class Banners:
    def __init__(self, flight_name='get_lucky'):
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
            cur.execute(f"CREATE TABLE flight_{flight_name}(x REAL, y REAL, z REAL, q REAL, u REAL, a REAL, t REAL, r_save_loc TEXT, n_save_loc TEXT, f_save_loc TEXT)")
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
        # cur.execute(f"SELECT * FROM clicks_{self.flight_name}")
        # res = cur.fetchall()
        # for r in res:
        #     print(r)



class Relater:
    def __init__(self, banners_id):
        self.db_c = sqlite3.connect('drone_disease.db')
        self.ban_id = banners_id
        self.flight_name = banners_id
        cur = self.db_c.cursor()
        res = cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='relations_{self.flight_name}';")
        if len(res.fetchall()) == 0:
            cur.execute(f"CREATE TABLE relations_{self.flight_name} (img_loc TEXT, pred_pixel_x array, pred_pixel_y array, blob_center_x array, blob_center_y array)")
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
            # print("points", pts)
            if len(pts) > 0:
                img_pts_x = []
                img_pts_y = []
                for point in pts:
                    point = self.convert_to_local(pose, point)
                    # print(pts)
                    pxs = self.fProject(point, rgb_K)
                    # print(pxs[1,:])
                    img_pts_x.append(pxs[0,:][0])
                    img_pts_y.append(pxs[1,:][0])
                predictions.append((r[7], np.array(img_pts_x), np.array(img_pts_y)))
                # print(q)
        # print(predictions)
        cur.executemany(f"INSERT INTO relations_{self.flight_name} (img_loc, pred_pixel_x, pred_pixel_y) VALUES (?, ?, ?)", predictions)
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
    


class Analyzer:
    def __init__(self, flight_name, fov, root, bag, cs_v):
        self.flight_name = flight_name
        self.field_of_view = fov
        self.cooper = DBConnector()

        self.dir_files = Dir(root, flight_name)
        self.dir_files.bag = os.path.join(root, bag)
        self.dir_files.clicks = os.path.join(root, cs_v)


    def preprocess(self, datadir):
        TOPICS = ['/current_pose', '/cam0/nv12_decode_result', '/cam1/nv12_decode_result', '/therm/image_raw_throttle']
        B = Banners(flight_name=self.flight_name)
        B.dir = datadir
        if B.is_new:
            print("ripping bag")
            # print(cc.dir)
            # process each bag
            bag = rosbag.Bag(datadir.bag)
            for topic, msg, t in bag.read_messages(TOPICS):
                # stack 3 images together
                if topic == '/therm/image_raw_throttle':
                    B.flir = msg
                if topic == '/cam0/nv12_decode_result':
                    B.rgb = msg
                if topic == '/cam1/nv12_decode_result':
                    B.noir = msg
                if topic == "/current_pose":
                    B.pose = msg
                    # print('stacking')
                    B.stack(t)

            B.csv_read(datadir.clicks)

        return B
    

    def findRelations(self, cam_combo, re_run=False):
        if re_run == True:
            self.cooper.db_c.execute(f"DROP TABLE relations_{self.flight_name}")
        D = Relater(self.flight_name)
        D.clicks_to_image(self.field_of_view)


    def showTrajectory(self, plot):
        db_c = sqlite3.connect('drone_disease.db')
        cur = db_c.cursor()
        # get position of clicks
        res = cur.execute(f"SELECT x, y FROM clicks_{self.flight_name}")
        clicks = cur.fetchall()
        clicks = np.array(clicks)
        
        # get all odom points
        res = cur.execute(f"SELECT x, y FROM flight_{self.flight_name}")
        odom = cur.fetchall()
        odom = np.array(odom)
        # get related points
            # SELECT t1.name
            # FROM table1 t1
            # LEFT JOIN table2 t2 ON t2.name = t1.name
            # WHERE t2.name IS NULL

        res = cur.execute(f"select flight_{self.flight_name}.x, flight_{self.flight_name}.y from flight_{self.flight_name} inner join relations_{self.flight_name} on flight_{self.flight_name}.r_save_loc == relations_{self.flight_name}.img_loc")
        seen = cur.fetchall()
        seen = np.array(seen)
        # print(seen)

        # plot everything
        # print(odom[:,0], odom[:,1])
        plot.scatter(odom[:,0], odom[:,1], color='blue', label="raw position")
        plot.scatter(clicks[:,0], clicks[:,1], color='red', label="clicks")
        plot.scatter(seen[:,0], seen[:,1], color='orange', label="position see click")
        plot.legend()
        return 


    def prettyImage(self, cc):
        res = self.cooper.getFrom('img_loc', f"relations_{self.flight_name}", max=5)
        found_squares = []
        for row in res:
            # img_loc, pred_pixel_x, pred_pixel_y
            print(row)
            # rgb = cv2.imread(row[7]) # for using flight data raw
            rgb = cv2.imread(row[0])
            rgb = cv2.undistort(rgb, rgb_K, rgb_dist)
            rgb = cv2.flip(rgb, 0)

            frame, val, blur = self.getFrameTarget(rgb)
            fig, ax = plt.subplots(2, 2, figsize=(12, 7))
            # fig
            ax[0][0].imshow(frame)
            # f = f & np.array([val3, val3, val3]).swapaxes(0,2).swapaxes(0,1)
            # f = cv2.rectangle(f,tl, br,(255,255,255),7)
            ax[0][1].imshow(val)
            ax[1][0].imshow(blur)

            self.showTrajectory(ax[1][1])
            
            # ax[1][1].hist(blur[blur > 0].flatten(), 255)
            # ax[5].imshow(kp)

            plt.savefig(f'IMG_FOR_TEST_{time.time()}.png')  # may need png instead of eps
            plt.close()
            

    def findWithHSV(self, frame):

        max_value = 255
        max_value_H = 360//2
        low_H = 117
        low_S = 90
        low_V = 79
        high_H = 180
        high_S = 255
        high_V = 193
        # convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask1 = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
        # mask2 = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
        # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(frame,frame, mask= mask)

        # attrition of signal
        for _ in range(BLUR_ITER):
            mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

        mask1 = (mask1//5)**2

        for _ in range(BLUR_ITER):
            mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        mask1 = cv2.filter2D(mask1, -1, kernel)

        for _ in range(BLUR_ITER*2):
            mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        mask1 = cv2.filter2D(mask1, -1, kernel)

        mask1 = (mask1//3)**2

        for _ in range(BLUR_ITER*2):
            mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

        for _ in range(2):
            mask1 = (mask1//5)**2
            ret, mask1 = cv2.threshold(mask1,175,255,cv2.THRESH_BINARY)

            for _ in range(BLUR_ITER):
                mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)
        
        ret, mask1 = cv2.threshold(mask1,20,255,cv2.THRESH_BINARY)

        for _ in range(BLUR_ITER//3):
            mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

        return mask1


    def findSquare(self, frame):
        
        c, h = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(c)
        # a = cv2.contourArea(c)
        c.sort(key = len)
        # val3 = val2.copy()
        centroids = []
        # return frame, centroids

        for cont in c:
            approx = cv2.approxPolyDP(cont, 4, True)
            # if len(approx) < 18 and len(approx) > 14:
            # print('yoohoo!', approx)    
            frame = cv2.drawContours(image=frame, contours=[approx], contourIdx=-1, color=(255, 255, 0), thickness=10)
            # print(approx[0])
            frame = cv2.putText(frame, str(len(approx)), tuple(approx[0,0]), font, fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
            center = np.average(cont, axis=0)[0]
            # print(int(center[0]), int(center[1]))
            center = (int(center[0]), int(center[1]))
            centroids.append(center)
            frame = cv2.circle(frame, center, radius=15, color=(0, 255, 255), thickness=3)
        return frame, np.array(centroids)
    

    def getFrameTarget(self, frame):  # frame is cc_parsed
        # filter the image for targets
        filtered = self.findWithHSV(frame)
        # extract centroids for the targets
        highlighted, c = self.findSquare(filtered)
        # print(c)

        return highlighted, filtered, frame


    def computeCentroids(self, cc):
        db_c = sqlite3.connect('drone_disease.db')
        cur = db_c.cursor()
        res = cur.execute(f"SELECT img_loc FROM relations_{self.flight_name}")
        found_squares = []
        idx = 0
        for row in res.fetchall():
            if idx == 10:
                break
            # img_loc, pred_pixel_x, pred_pixel_y
            # print(row)
            # rgb = cv2.imread(row[7]) # for using flight data raw
            rgb = cv2.imread(row[0])
            rgb = cv2.undistort(rgb, rgb_K, rgb_dist)
            rgb = cv2.flip(rgb, 0)

            filtered = self.findWithHSV(rgb)
            # extract centroids for the targets
            highlighted, c = self.findSquare(filtered)
            if c.size == 0:
                c = np.array([[-1, -1]])
            # print(c)
            # thing = (c[:,0], c[:,1], row[0])
            thing = (c[:,0], c[:,1], row[0])
            found_squares.append(thing)
            idx += 1

            # compute deltas from c to pred pixels
                # zip ppx and ppy

        cur.executemany(f"UPDATE relations_{self.flight_name} SET blob_center_x = ?, blob_center_y = ? WHERE img_loc = ?;", found_squares)
        db_c.commit()


    def computeRangeAndBearings(self):
        # get all predicted points and all detected points 
        p_and_d = self.cooper.getFrom('pred_pixel_x, pred_pixel_y, blob_center_x, blob_center_x', f'relations_{self.flight_name}')

        errors = []
        for e in p_and_d:
            # get range between points
            # print(type(e[0]))
            pred_xy = np.column_stack((e[0], e[1]))
            # print('pred', pred_xy)
            foun_xy = np.column_stack((e[2], e[3]))
            # print('found', foun_xy)
            if foun_xy[0,0] != None:
                optimal = []
                for p in pred_xy:
                    # print(p, foun_xy)
                    diff = foun_xy - p
                    # print(diff)
                    euclid = np.abs(np.linalg.norm(diff, 2, axis=0))
                    bearin = (np.arctan2(diff[:,0], diff[:,1]) * 180 / np.pi)
                    min_idx = np.argmin(euclid)
                    best = [euclid[min_idx], bearin[min_idx]]
                    optimal.append(best)
                    # print('best measurments', best)
                    # print('range', euclid)
                    # print('bearing', bearin)
                errors.append(optimal)
        # print('all of the errors', errors)
        print('average distance error', np.average(np.array(errors)[:,:,0]))
        # TODO: actually generate a report for the data


    # TODO: this
    def genReport(self, data):        
        pass


    def doTheRoar(self):
        cam_com = self.preprocess(self.dir_files)
        # record projected points
        self.findRelations(cam_com)
        self.computeCentroids(cam_com)
        self.computeRangeAndBearings()


    def verifyFlight(self):
        cam_com = self.preprocess(self.dir_files)
        # record projected points
        self.findRelations(cam_com)
        self.prettyImage(cam_com)


