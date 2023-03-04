from banners import *
from data_relater import *
import cv2
import matplotlib.pyplot as plt
import time
import pickle

HIGH = 1
THRESHOLD1 = 0.7
THRESHOLD1 *= 255
THRESHOLD1 = int(THRESHOLD1)
THRESHOLD2 = 0.8
THRESHOLD2 *= 255
THRESHOLD2 = int(THRESHOLD2)
BLUR_ITER = 3
BLUR_KERNEL = (5,5)


GLOWUP = 275

def cleanStacks():
    dir = "calibrate/stacks/"
    for f in os.listdir(dir):
        if ".bin" in f:
            os.remove(os.path.join(dir, f))


# input      bag and csv 
# return     banners.cameraCombo and banners.csv_augmented
# # this gets the raw data in
def preprocess(datadir):
    TOPICS = ['/current_pose', '/cam0/nv12_decode_result', '/cam1/nv12_decode_result', '/therm/image_raw_throttle']
    cc = CameraCombo()
    cc.dir = datadir
    # print(cc.dir)
    # process each bag
    bag = rosbag.Bag(datadir.bag)
    for topic, msg, t in bag.read_messages(TOPICS):
        # stack 3 images together
        if topic == '/therm/image_raw_throttle':
            cc.flir = msg
        if topic == '/cam0/nv12_decode_result':
            cc.rgb = msg
        if topic == '/cam1/nv12_decode_result':
            cc.noir = msg
        if topic == "/current_pose":
            cc.pose = msg
            cc.stack(t)
    cc.done()

    C = CSVAugmented(datadir.clicks, datadir.path)
    C.csv_read()
    C.done()

    return cc, C


def restart(datadir):
    cc = CameraCombo()
    cc.dir = datadir
    cc.parsed = pd.read_pickle(os.path.join(datadir.path, "parsed.pkl"))

    C = CSVAugmented(datadir.clicks, datadir.path)
    C.click_data = pd.read_pickle(os.path.join(datadir.path, "related.pkl"))


    return cc, C


# this relates the click data to image space
# it reurns a dataframe 
def findRelations(cam_combo, cs_v):
    D = Relater()
    D.cc = cam_combo
    return D.clicks_to_image(cs_v, 30)

# input      list of stacks
# return     the stack name with a computed blob center 
# # this pulls the target from individual frames with blob detection
def getFrameTarget(frame):  # frame is cc_parsed
    # print(f'frame.shape: {frame.shape}')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    for _ in range(BLUR_ITER):
        cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        # cv2.threshold(frame, THRESHOLD2, HIGH, cv2.THRESH_BINARY)
    
    ret, val = cv2.threshold(frame, THRESHOLD1, HIGH, cv2.THRESH_BINARY)
    val2 = val.copy()
    # print(f'val.shape: {val.shape}') #1920x1080 -> 1036x1888
    for _ in range(BLUR_ITER):
        cv2.GaussianBlur(val2, BLUR_KERNEL, 0)
        cv2.threshold(val2, THRESHOLD2, HIGH, cv2.THRESH_BINARY)
    # val = cv2.Canny(val, 30, 200)  # an alternative to binary thresholding
    c, h = cv2.findContours(val2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(c)
    val3 = val2.copy()
    cv2.drawContours(image=val3, contours=c, contourIdx=-1, hierarchy=h, color=(255, 0, 0), thickness=50)
    return frame, val, val2, val3


def prettyImage(cc, filename):
    centers = pd.DataFrame(columns=['stack_loc', 'centroids'])
    for idx, row in cc.parsed.iterrows():
        if idx > GLOWUP:
            print(row['save_loc'])
            with open(row['save_loc'], 'rb') as f:
                stack = pickle.load(f)
                rgb = bridge.imgmsg_to_cv2(stack['rgb'], desired_encoding='passthrough')
                rgb = cc.undistort(rgb, rgb_K, rgb_dist)
                rgb = cv2.flip(rgb, 0)

                frame, val, val2, val3 = getFrameTarget(rgb)
                fig, ax = plt.subplots(2, 2)
                ax[0][0].imshow(frame)
                ax[0][1].imshow(val)
                # ax[0][2].imshow(blur)
                ax[1][0].imshow(val2)
                ax[1][1].imshow(val3)
                # ax[5].imshow(kp)
                plt.savefig(f'{filename}{time.time()}.png')  # may need png instead of eps
                plt.close()
    #         centers.loc[len(centers.index)] = [row['stack_location'], blob_center]
    # return centers



def readImages(cc):
    centers = pd.DataFrame(columns=['stack_loc', 'centroids'])
    for idx, row in cc.parsed.iterrows():
        with open(row['save_loc'], 'rb') as f:
            stack = pickle.load(f)
            rgb = bridge.imgmsg_to_cv2(stack['rgb'], desired_encoding='passthrough')
            rgb = cc.undistort(rgb, rgb_K, rgb_dist)
            rgb = cv2.flip(rgb, 0)
            blob_center = getFrameTarget(rgb)
            centers.loc[len(centers.index)] = [row['save_loc'], blob_center]
    return centers


# this joins the blob data with predicted data a
def joinAndCompute(rel, blob, maxval=HIGH):
    
    pass


def setup():
    root = "calibrate"
    dir_files = Dir(root, "test")
    dir_files.bag = os.path.join(root, "rtk-co-locate.bag")
    dir_files.clicks = os.path.join(root, "rtk-co-locate-click.csv")
    # check for pickles
    if 'parsed.pkl' not in os.listdir(root):
        cleanStacks()
    
    # print(dir_files)
    return dir_files


def main():

    # setup
    dir_files = setup()

    # read in data    
    # if 'parsed.pkl' in os.listdir(dir_files.path):
    #     cam_com, c_s_v = restart(dir_files)
    # else:
    cam_com, c_s_v = preprocess(dir_files)

    # record projected points
    relations = findRelations(cam_com, c_s_v)
    
    # find truth 
    # truth = readImages(cam_com)
    prettyImage(cam_com, f'datagen_20230301_')

    # compare predictions to blobs
    # joinAndCompute(relations, truth)



main()