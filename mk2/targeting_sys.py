# rm *.png; python3 mk2/targeting_sys.py


from banners_mk2 import *
# from data_relater import *
import cv2
import matplotlib.pyplot as plt
import time
import pickle


FIELD_OF_VIEW = 20
TESTING_NUM = 5

HIGH = 1
THRESHOLD1 = 0.25
THRESHOLD1 *= 255
THRESHOLD1 = int(THRESHOLD1)
THRESHOLD2 = 0.9
THRESHOLD2 *= 255
THRESHOLD2 = int(THRESHOLD2)
BLUR_ITER = 8
BLUR_KERNEL = (15,15)
BLUR_GAUSSVAR = 10  # Very important hyperparameter
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 3
# Line thickness of 2 px
thickness = 5

cooper = DBConnector()


# input      bag and csv 
# return     banners.cameraCombo and banners.csv_augmented
# # this gets the raw data in
def preprocess(datadir):
    TOPICS = ['/current_pose', '/cam0/nv12_decode_result', '/cam1/nv12_decode_result', '/therm/image_raw_throttle']
    B = Banners()
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


# this relates the click data to image space
# it reurns a dataframe 
def findRelations(cam_combo, re_run=False):
    if re_run == True:
        cooper.db_c.execute("DROP TABLE relations_get_lucky")
    D = Relater(cam_combo.flight_name)
    D.clicks_to_image(FIELD_OF_VIEW)


def showTrajectory(plot):
    db_c = sqlite3.connect('drone_disease.db')
    cur = db_c.cursor()
    # get position of clicks
    res = cur.execute(f"SELECT x, y FROM clicks_{'get_lucky'}")
    clicks = cur.fetchall()
    clicks = np.array(clicks)
    
    # get all odom points
    res = cur.execute(f"SELECT x, y FROM flight_{'get_lucky'}")
    odom = cur.fetchall()
    odom = np.array(odom)
    # get related points
        # SELECT t1.name
        # FROM table1 t1
        # LEFT JOIN table2 t2 ON t2.name = t1.name
        # WHERE t2.name IS NULL

    res = cur.execute(f"select flight_get_lucky.x, flight_get_lucky.y from flight_get_lucky inner join relations_get_lucky on flight_get_lucky.r_save_loc == relations_get_lucky.img_loc")
    seen = cur.fetchall()
    seen = np.array(seen)
    # print(seen)

    # plot everything
    # print(odom[:,0], odom[:,1])
    plot.scatter(odom[:,0], odom[:,1], color='blue', lebel="raw position")
    plot.scatter(clicks[:,0], clicks[:,1], color='red', label="clicks")
    plot.scatter(seen[:,0], seen[:,1], color='orange', label="position see click")
    plot.legend()
    return 


def findWithHSV(frame):

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


def theGreatFilterEvent(frame):
    # get the red color channel
    mask = findWithHSV(frame)
    
    return mask, frame


def findSquare(frame):
    
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


# input      list of stacks
# return     the stack name with a computed blob center 
# # this pulls the target from individual frames with blob detection
def getFrameTarget(frame):  # frame is cc_parsed
    # filter the image for targets
    filtered, frame = theGreatFilterEvent(frame)
    # extract centroids for the targets
    highlighted, c = findSquare(filtered)
    print(c)

    return highlighted, filtered, frame


def prettyImage(cc):
    db_c = sqlite3.connect('drone_disease.db')
    cur = db_c.cursor()
    res = cur.execute(f"SELECT img_loc FROM relations_{'get_lucky'} LIMIT 10")
    idx = 0
    found_squares = []
    for row in res.fetchall():
        if idx > -1:
            # img_loc, pred_pixel_x, pred_pixel_y
            print(row)
            # rgb = cv2.imread(row[7]) # for using flight data raw
            rgb = cv2.imread(row[0])
            rgb = cv2.undistort(rgb, rgb_K, rgb_dist)
            rgb = cv2.flip(rgb, 0)

            frame, val, blur = getFrameTarget(rgb)
            fig, ax = plt.subplots(2, 2, figsize=(12, 7))
            # fig
            ax[0][0].imshow(frame)
            # f = f & np.array([val3, val3, val3]).swapaxes(0,2).swapaxes(0,1)
            # f = cv2.rectangle(f,tl, br,(255,255,255),7)
            ax[0][1].imshow(val)
            ax[1][0].imshow(blur)

            showTrajectory(ax[1][1])
            
            # ax[1][1].hist(blur[blur > 0].flatten(), 255)
            # ax[5].imshow(kp)

            plt.savefig(f'IMG_FOR_TEST_{time.time()}.png')  # may need png instead of eps
            plt.close()
        idx += 1
    

    # found_squares = [f for f in found_squares if f > 0 and f < 100000]
    # plt.hist(found_squares)
    # plt.show()

    #         centers.loc[len(centers.index)] = [row['stack_location'], blob_center]
    # return centers


def computeCentroids(cc):
    db_c = sqlite3.connect('drone_disease.db')
    cur = db_c.cursor()
    res = cur.execute(f"SELECT img_loc FROM relations_{'get_lucky'}")
    found_squares = []
    idx = 0
    for row in res.fetchall():
        if idx == 10:
            break
        # img_loc, pred_pixel_x, pred_pixel_y
        print(row)
        # rgb = cv2.imread(row[7]) # for using flight data raw
        rgb = cv2.imread(row[0])
        rgb = cv2.undistort(rgb, rgb_K, rgb_dist)
        rgb = cv2.flip(rgb, 0)

        filtered, frame = theGreatFilterEvent(rgb)
        # extract centroids for the targets
        highlighted, c = findSquare(filtered)
        if c.size == 0:
            c = np.array([[-1, -1]])
        print(c)
        # thing = (c[:,0], c[:,1], row[0])
        thing = (c[:,0], c[:,1], row[0])
        found_squares.append(thing)
        idx += 1

        # compute deltas from c to pred pixels
            # zip ppx and ppy

    cur.executemany(f"UPDATE relations_get_lucky SET blob_center_x = ?, blob_center_y = ? WHERE img_loc = ?;", found_squares)
    db_c.commit()
        

def computeRangeAndBearings():
    # get all predicted points and all detected points 
    p_and_d = cooper.getFrom('pred_pixel_x, pred_pixel_y, blob_center_x, blob_center_x', 'relations_get_lucky', max=TESTING_NUM)

    errors = []
    for e in p_and_d:
        # get range between points
        # print(type(e[0]))
        pred_xy = np.column_stack((e[0], e[1]))
        # print('pred', pred_xy)
        foun_xy = np.column_stack((e[2], e[3]))
        # print('found', foun_xy)
        optimal = []
        for p in pred_xy:
            diff = foun_xy - p
            # print(diff)
            euclid = np.abs(np.linalg.norm(diff, 2, axis=0))
            bearin = (np.arctan2(diff[:,0], diff[:,1]) * 180 / np.pi)
            min_idx = np.argmin(euclid)
            best = [euclid[min_idx], bearin[min_idx]]
            optimal.append(best)
            print('best measurments', best)
            # print('range', euclid)
            # print('bearing', bearin)
        errors.append(optimal)



    print('all of the errors', errors)
    print('average error', np.average(np.array(errors)[:,:,0]))


def setup():
    root = "calibrate"
    dir_files = Dir(root, "test")
    dir_files.bag = os.path.join(root, "rtk-co-locate.bag")
    dir_files.clicks = os.path.join(root, "rtk-co-locate-click.csv")
    
    # print(dir_files)
    return dir_files


def main():

    # cooper.setupArrays()
    # setup
    dir_files = setup()

    cam_com = preprocess(dir_files)

    # record projected points
    findRelations(cam_com, False)
    cooper.diagnostic('relations_get_lucky', max=TESTING_NUM)
    # find truth 
    # truth = readImages(cam_com)
    prettyImage(cam_com)
    computeCentroids(cam_com)

    print('\n\n===============================================================================================================\n\n')

    cooper.diagnostic('relations_get_lucky', max=TESTING_NUM)
    # compare predictions to blobs
    computeRangeAndBearings()
    # joinAndCompute(relations, truth)



main()