# backup of the big complicated filter
def big_f():
    # Upon the arrival of the tri solarians...
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

    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    # filter2D() function can be used to apply kernel to an image.
    # Where ddepth is the desired depth of final image. ddepth is -1 if...
    # ... depth is same as original or source image.
    identity = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel1)
    


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
    # print(mean, sd)

    frame[frame < mean + (3*sd)] = 0

    counts, bins = np.histogram(frame.copy().flatten(), 255)
    mids = 0.5*(bins[1:] + bins[:-1])
    probs = counts / np.sum(counts)

    mean = np.sum(probs * mids)  
    sd = np.sqrt(np.sum(probs * (mids - mean)**2))
    # print(blur_data)
    # print(mean, sd)

    frame[frame < mean + (1*sd)] = 0

    # _, frame = cv2.threshold(frame, THRESHOLD2, HIGH, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, BLUR_KERNEL)
    cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel, iterations=BLUR_ITER)
    
    first_blur = frame.copy()

    frame = (frame//2)**1.8
    for _ in range(BLUR_ITER):
        frame = cv2.GaussianBlur(frame, BLUR_KERNEL, BLUR_GAUSSVAR)
        # _, frame = cv2.threshold(frame, THRESHOLD2, HIGH, cv2.THRESH_BINARY)

    # TODO: do this loop:: edge detect, contours * crank, blur
        # this will favor the square 
        # irregular shapes will have less favorable ratios of white to black and can be erroded with a blur
        
    # print(type(frame[0,0]))

    frame *= 255.0/frame.max()
    frame = frame.astype('uint8')


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



#===========================================================
#
#
#=============================================================



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

    mask1 = (mask1//3)**2

    for _ in range(BLUR_ITER):
        mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    mask1 = cv2.filter2D(mask1, -1, kernel)

    for _ in range(BLUR_ITER*2):
        mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

    mask1 = (mask1//3)**2

    for _ in range(BLUR_ITER*2):
        mask1 = cv2.GaussianBlur(mask1, BLUR_KERNEL, BLUR_GAUSSVAR)

    return mask1


def findWithBlur(frame):
    red = frame[:,:,2]
    for _ in range(BLUR_ITER):
        red = cv2.GaussianBlur(red, BLUR_KERNEL, BLUR_GAUSSVAR)
    # red = (red//4)**2

    mask = red
    return mask


def theGreatFilterEvent(frame):
    # get the red color channel
    mask = findWithHSV(frame)
    
    return mask, frame


def findSquare(frame):
    centroid = []
    c, h = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(c)
    # a = cv2.contourArea(c)
    c.sort(key = len)
    # val3 = val2.copy()
    centroids = []
    for cont in c:
        approx = cv2.approxPolyDP(cont, 4, True)
        if len(approx) < 18 and len(approx) > 14:
            # print('yoohoo!', approx)    
            frame = cv2.drawContours(image=frame, contours=[approx], contourIdx=-1, color=(255, 255, 0), thickness=10)
            # print(approx[0])
            frame = cv2.putText(frame, str(len(approx)), tuple(approx[0,0]), font, fontScale, (255, 255, 0), thickness, cv2.LINE_AA)
            center = np.average(cont, axis=0)[0]
            # print(int(center[0]), int(center[1]))
            center = (int(center[0]), int(center[1]))
            centroids.append(center)
            frame = cv2.circle(frame, center, radius=15, color=(0, 255, 255), thickness=3)
    return frame, centroids


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




# ==================================================================
# 
# 
# ==================================================================


inline void convertYUVtoRGB8Pixel(int y, int u, int v, uint8_t& r, uint8_t& g, uint8_t& b) {
    // numbers from http://paulbourke.net/dataformats/nv12/
    int tmp_r = y + (int)(1.402f*v); 
    int tmp_g = y - (int)(0.714f*v + 0.344f*u);
    int tmp_b = y + (int)(1.772f*u);
    r = tmp_r>255? 255 : tmp_r<0 ? 0 : tmp_r;
    g = tmp_g>255? 255 : tmp_g<0 ? 0 : tmp_g;
    b = tmp_b>255? 255 : tmp_b<0 ? 0 : tmp_b; 
    // Having trouble with BGR-ish output in rviz, 
    // where fiddling with the magic numbers doesn't seem to have much impact...
     
}

void convertYUV420SPtoRGB8Image(uint8_t* yuv420, uint8_t* rgb8, int width, int height, int uv_offset) {
    assert(yuv420 && "Nullpointer as input");
    assert(rgb8 && "Nullpointer as output");

    size_t pixel_size = width*height;
    //uv is stored below the luminance image: add offset
    size_t offset = pixel_size + uv_offset; 
    
    for(size_t y=0, uv=0; y < pixel_size; y+=2, uv+=2) {

      int v = yuv420[offset + uv  ] & 0xff;
      v -= 128;
      int u = yuv420[offset + uv+1] & 0xff;
      u -= 128;
      
      for(int i : { 0, 1, width, width+1 }) {//Walk through pixels belonging to uv block
        size_t pix_rgb = 3*(y + i); //location of rgb pixel
        int luma_val = yuv420[y + i] & 0xff;
        convertYUVtoRGB8Pixel(luma_val, u, v, rgb8[pix_rgb], rgb8[pix_rgb+1], rgb8[pix_rgb+2]);
      }

      if (y!=0 && (y+2)%width==0){//End of line
        y += width; //Skip one line in input b/c uv blocks are 2x2
      }
    }
}


def convertYUVtoRGB8Pixel(y, u, v):
    tmp_r = y + int(1.402*v) 
    tmp_g = y - int(0.714*v + 0.344*u)
    tmp_b = y + int(1.772*u)
    r = max(0, min(tmp_r, 255))
    g = max(0, min(tmp_g, 255))
    b = max(0, min(tmp_b, 255))
    return r, g, b

def convertYUV420SPtoRGB8Image(yuv, width, height, uv_offset):
    pixel_size = width * height
    offset = pixel_size + uv_offset

    rgb = np.array(width * height * 3)

    for y, uv in range(0, pixel_size, 2):
        v = yuv[offset + uv] & 0xff
        v -= 128
        u = yuv[offset + uv+1] & 0xff
        u -= 128

        for i in [0, 1, width, width+1]:
            # location of rgb pixel
            pix_rgb = 3 * (y + i)
            luma_val = yuv[y + i] & 0xff
            rgb[pix_rgb], rgb[pix_rgb+1], rgb[pix_rgb+2] = convertYUVtoRGB8Pixel(luma_val, u, v)

        if y!=0 and (y+2) % width == 0:
            y += width

    return rgb