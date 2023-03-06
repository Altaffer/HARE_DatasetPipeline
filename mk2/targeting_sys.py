from banners_mk2 import *
# from data_relater import *
import cv2
import matplotlib.pyplot as plt
import time
import pickle

HIGH = 1
THRESHOLD1 = 0.25
THRESHOLD1 *= 255
THRESHOLD1 = int(THRESHOLD1)
THRESHOLD2 = 0.9
THRESHOLD2 *= 255
THRESHOLD2 = int(THRESHOLD2)
BLUR_ITER = 8
BLUR_KERNEL = (15,15)
BLUR_GAUSSVAR = 1  # Very important hyperparameter

template = cv2.imread('mk2/Screenshot from 2023-03-02 22-26-11.png')
GLOWUP = 275
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 3
# Line thickness of 2 px
thickness = 5


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
    return D.clicks_to_image(25)


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
    frame = frame.astype('float64')
    # print(frame.shape)
    frame = frame[:,:,0] * frame[:,:,1] * frame[:,:,2]
    peaky = frame.copy()
    frame *= 255.0/frame.max()
    frame = frame.astype('uint8')
    
    # print(frame.shape)
    # print(type(frame[0,0]))


    # print(f'val.shape: {val.shape}') #1920x1080 -> 1036x1888
    for _ in range(BLUR_ITER):
        frame = cv2.GaussianBlur(frame, BLUR_KERNEL, BLUR_GAUSSVAR)
        # _, frame = cv2.threshold(frame, THRESHOLD2, HIGH, cv2.THRESH_BINARY)
        # _, val = cv2.Canny(val, 30, 200)  # an alternative to binary thresholding

    blur_data = frame.copy().flatten()
    counts, bins = np.histogram(blur_data, 255)
    mids = 0.5*(bins[1:] + bins[:-1])
    probs = counts / np.sum(counts)

    mean = np.sum(probs * mids)  
    sd = np.sqrt(np.sum(probs * (mids - mean)**2))
    # print(blur_data)
    print(mean, sd)

    frame[frame < mean + (3*sd)] = 0

    counts, bins = np.histogram(frame.copy().flatten(), 255)
    mids = 0.5*(bins[1:] + bins[:-1])
    probs = counts / np.sum(counts)

    mean = np.sum(probs * mids)  
    sd = np.sqrt(np.sum(probs * (mids - mean)**2))
    # print(blur_data)
    print(mean, sd)

    frame[frame < mean + (1*sd)] = 0

    # _, frame = cv2.threshold(frame, THRESHOLD2, HIGH, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, BLUR_KERNEL)
    cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel, iterations=BLUR_ITER)
    
    first_blur = frame.copy()

    for _ in range(BLUR_ITER):
        frame = (frame//2)**2
        frame = cv2.GaussianBlur(frame, BLUR_KERNEL, BLUR_GAUSSVAR)
        # _, frame = cv2.threshold(frame, THRESHOLD2, HIGH, cv2.THRESH_BINARY)
        


    # fit a double gaussian and the higher value smaller hump is the thing you are looking for
     
    # val = cv2.cvtColor(val, cv2.COLOR_RGB2GRAY)
    c, h = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(c)
    # a = cv2.contourArea(c)
    c.sort(key = len)
    # val3 = val2.copy()
    for cont in c:
        approx = cv2.approxPolyDP(cont, 4, True)
        if len(approx) == 8:
            # print('yoohoo!', approx)    
            frame = cv2.drawContours(image=frame, contours=[approx], contourIdx=-1, color=(255, 255, 0), thickness=10)
            # print(approx[0])
            frame = cv2.putText(frame, 'this one!', tuple(approx[0,0]), font, fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
    # approx = cv2.approxPolyDP(c[1], 5, True)    
    # frame = cv2.drawContours(image=frame, contours=[approx], contourIdx=-1, color=(255, 255, 0), thickness=10)
    # frame = cv2.putText(frame, 'this is the one', tuple(approx[0,0]), font, fontScale, (255, 255, 0), thickness, cv2.LINE_AA)


    # frame = cv2.drawContours(image=frame, contours=c, contourIdx=-1, hierarchy=h, color=(255, 0, 0), thickness=10)

    return frame, peaky, first_blur, blur_data


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
            rgb = cv2.imread(row[0])
            rgb = cv2.undistort(rgb, rgb_K, rgb_dist)
            rgb = cv2.flip(rgb, 0)

            frame, val, blur, data = getFrameTarget(rgb)
            # found_squares += a
            # f = rgb & np.array([val3, val3, val3]).swapaxes(0,2).swapaxes(0,1)
            # f, tl, br = findSquare(rgb)
            # plt.imshow(frame)
            fig, ax = plt.subplots(2, 2, figsize=(12, 7))
            # fig
            ax[0][0].imshow(frame)
            # f = f & np.array([val3, val3, val3]).swapaxes(0,2).swapaxes(0,1)
            # f = cv2.rectangle(f,tl, br,(255,255,255),7)
            ax[0][1].imshow(val)
            ax[1][0].imshow(blur)
            
            ax[1][1].hist(blur[blur > 0].flatten(), 255)
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