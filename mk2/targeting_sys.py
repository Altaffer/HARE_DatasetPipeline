from banners_mk2 import *
# from data_relater import *
import cv2
import matplotlib.pyplot as plt
import time
import pickle

HIGH = 1
THRESHOLD1 = 0.75
THRESHOLD1 *= 255
THRESHOLD1 = int(THRESHOLD1)
THRESHOLD2 = 0.9
THRESHOLD2 *= 255
THRESHOLD2 = int(THRESHOLD2)
BLUR_ITER = 4
BLUR_KERNEL = (9,9)
BLUR_GAUSSVAR = 0.5  # Very important hyperparameter

template = cv2.imread('mk2/Screenshot from 2023-03-02 22-26-11.png')
GLOWUP = 275



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
                B.stack(t)

        B.csv_read(datadir.clicks)

    return B


# this relates the click data to image space
# it reurns a dataframe 
def findRelations(cam_combo):
    D = Relater(cam_combo.flight_name)
    return D.clicks_to_image(30)


# input      list of stacks
# return     the stack name with a computed blob center 
# # this pulls the target from individual frames with blob detection
def getFrameTarget(frame):  # frame is cc_parsed
    # print(f'frame.shape: {frame.shape}')
    # normalize frame brightness
    alpha = 1.25
    beta = 0
    
    # frame = cv2.convertScaleAbs(frame, alpha = alpha, beta = beta)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    val = frame
    
    # print(type(frame))
    # print(type(frame[0,0]))


    # print(f'val.shape: {val.shape}') #1920x1080 -> 1036x1888
    for _ in range(BLUR_ITER):
        val = cv2.GaussianBlur(id(val), BLUR_KERNEL, BLUR_GAUSSVAR)
        val = cv2.threshold(id(val), THRESHOLD2, HIGH, cv2.THRESH_BINARY)
        # val = cv2.Canny(val, 30, 200)  # an alternative to binary thresholding

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, BLUR_KERNEL)
    cv2.morphologyEx(id(val), cv2.MORPH_GRADIENT, kernel, iterations=BLUR_ITER)

    for _ in range(BLUR_ITER):
        val = cv2.GaussianBlur(id(val), BLUR_KERNEL, BLUR_GAUSSVAR)
        val = cv2.threshold(id(val), THRESHOLD2, HIGH, cv2.THRESH_BINARY)
        
    # val = cv2.cvtColor(val, cv2.COLOR_RGB2GRAY)
    c, h = cv2.findContours(id(val), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(c)
    # a = cv2.contourArea(c)
    # val3 = val2.copy()
    # for cont in c:
    #     approx = cv2.approxPolyDP(cont, 5, True)    
    val = cv2.drawContours(image=val, contours=c, contourIdx=-1, hierarchy=h, color=(255, 0, 0), thickness=10)

    return val


def findSquare(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter using threshold area
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print(cnts)

    min_area = 100
    max_area = 2000
    image_number = 0
    areas = []
    for c in cnts:
        area = cv2.contourArea(c)
        print(area)
        areas.append(area)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = frame[y:y+h, x:x+w]
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 15)
            print('square?')
            image_number += 1           

    return sharpen, close, thresh, frame, areas


def prettyImage(cc):
    db_c = sqlite3.connect('drone_disease.db')
    cur = db_c.cursor()
    res = cur.execute(f"SELECT img_loc FROM relations_{'get_lucky'}")
    idx = 0
    found_squares = []
    for row in res.fetchall():
        if idx > -1:
            # img_loc, pred_pixel_x, pred_pixel_y
            print(row)
            # rgb = cv2.imread(row[7]) # for using flight data raw
            rgb = cv2.imread(row[0], 0)
            rgb = cv2.undistort(rgb, rgb_K, rgb_dist)
            rgb = cv2.flip(rgb, 0)

            frame = getFrameTarget(rgb)
            # found_squares += a
            # f = rgb & np.array([val3, val3, val3]).swapaxes(0,2).swapaxes(0,1)
            # f, tl, br = findSquare(rgb)
            plt.imshow(frame)
            # fig, ax = plt.subplots(2, 2, figsize=(12, 7))
            # # fig
            # ax[0][0].imshow(frame)
            # # f = f & np.array([val3, val3, val3]).swapaxes(0,2).swapaxes(0,1)
            # # f = cv2.rectangle(f,tl, br,(255,255,255),7)
            # ax[0][1].imshow(val)
            # # ax[0][2].imshow(blur)
            # ax[1][0].imshow(val2)
            # ax[1][1].imshow(val3)
            # ax[5].imshow(kp)

            plt.savefig(f'IMG_FOR_TEST_{time.time()}.png')  # may need png instead of eps
            plt.close()
        idx += 1

    # found_squares = [f for f in found_squares if f > 0 and f < 100000]
    # plt.hist(found_squares)
    # plt.show()

    #         centers.loc[len(centers.index)] = [row['stack_location'], blob_center]
    # return centers


# this joins the blob data with predicted data a
def joinAndCompute(rel, blob, maxval=HIGH):
    
    pass


def setup():
    root = "calibrate"
    dir_files = Dir(root, "test")
    dir_files.bag = os.path.join(root, "rtk-co-locate.bag")
    dir_files.clicks = os.path.join(root, "rtk-co-locate-click.csv")
    
    # print(dir_files)
    return dir_files


def main():

    # setup
    dir_files = setup()

    cam_com = preprocess(dir_files)

    # record projected points
    findRelations(cam_com)
    
    # find truth 
    # truth = readImages(cam_com)
    prettyImage(cam_com)

    # compare predictions to blobs
    # joinAndCompute(relations, truth)



main()