import cv2
import numpy as np
import tensorflow.keras
import sys
import matplotlib.pyplot as plt
from PIL import Image
import time
import serial

np.set_printoptions(suppress=True)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('C:/img/keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
try:
    ser = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(1)
except:
    print("Device can not be found or can not be configured.")
    sys.exit(0)

def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([20, 150, 20])
    upper = np.array([255, 255, 255])

    yellow_lower = np.array([0, 85, 81])
    yellow_upper = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)

    return masked

def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array(
        [[int(0.13*x), int(0.92*y)], [int(0.13*x), int(0.1*y)], [int(0.4*x), int(0.1*y)], [int(0.4*x), int(0.78*y)], [int(0.42*x), int(0.78*y)], [int(0.42*x), int(0.1*y)],[int(0.9*x), int(0.1*y)], [int(0.9*x), int(y)], [int(0.13*x), int(0.92*y)]])

    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])

    source = np.float32([[348, 300], [600, 300], [170, 500], [600, 500]])
    destination = np.float32([[w*0.2, 0], [w*0.8, 0], [w*0.2, h], [w*0.8, h]])



    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(destination, source)
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))

    return _image, minv

def plothistogram(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 4
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        # cv2.imshow("oo", out_img)

        if len(good_left) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
    rtx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]


    ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    return ret

def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return pts_mean, result
i=0
src = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(1)
#dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.4, interpolation=cv2.INTER_LINEAR)

while(src.isOpened()):
    i+=1
    retval, img = src.read()
    _, img_color = cap.read()
    #img_color = cv2.flip(t_cam, 1)
    frame = cv2.resize(img_color, (224, 224))
    img = 255-img
    dst = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_AREA) #######tae
    frame_array = np.asarray(frame)
    normalized_frame_array = (frame_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_frame_array
    hi = model.predict(data)
    # print(hi)
    stop = hi[0, 0]
    v20 = hi[0, 1]
    nosign = hi[0, 2]
    detected = max(stop, v20, nosign)
    if ((detected == stop)*(i%5==0)):
        print('stop detected.') #stop(s)
        ser.write(b's')
    if ((detected == v20)*(i%5==0)):
        print('velocity 20')  #20km(t)
        ser.write(b't')
    if detected == nosign:
        print('nosign') #nosign
    if cv2.waitKey(1) == ord('q'):
        break

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    hue_red = 0  # Red에 대한 hue값 설정
    lower_red = (hue_red - 10, 150, 0)
    upper_red = (hue_red + 10, 255, 255)

    hue_blue = 120  # Blue에 대한 hue값 설정
    lower_blue = (hue_blue - 20, 150, 0)
    upper_blue = (hue_blue + 20, 255, 255)

    img_mask_r = cv2.inRange(img_hsv, lower_red, upper_red)
    img_mask_b = cv2.inRange(img_hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_mask_r = cv2.morphologyEx(img_mask_r, cv2.MORPH_DILATE, kernel, iterations=3)  # 모폴로지 팽창을 통한 비어있는 부분 채우기
    img_mask_b = cv2.morphologyEx(img_mask_b, cv2.MORPH_DILATE, kernel, iterations=3)

    nlabels_r, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(img_mask_r)
    nlabels_b, labels_b, stats_b, centroids_b = cv2.connectedComponentsWithStats(img_mask_b)

    max_r = -1
    max_b = -1
    max_index_r = -1
    max_index_b = -1

    for x in range(nlabels_r):

        if x < 1:
            continue

        area_red = stats_r[x, cv2.CC_STAT_AREA]

        if area_red > 10000:  # 10000이상 범위로 검출시 신호 전송
            max_r = area_red
            max_index_r = x
            ser.write(b'r')
            print("빨강") #송신 r

    for y in range(nlabels_b):

        if y < 1:
            continue

        area_blue = stats_b[y, cv2.CC_STAT_AREA]

        if area_blue > 10000:  # 10000이상 범위로 검출시 신호 전송
            max_b = area_blue
            max_index_b = y
            ser.write(b'b')
            print("파랑") #송신 b

    if max_index_r != -1:
        center_x = int(centroids_r[max_index_r, 0])
        center_y = int(centroids_r[max_index_r, 1])
        left = stats_r[max_index_r, cv2.CC_STAT_LEFT]
        top = stats_r[max_index_r, cv2.CC_STAT_TOP]
        width = stats_r[max_index_r, cv2.CC_STAT_WIDTH]
        height = stats_r[max_index_r, cv2.CC_STAT_HEIGHT]

        cv2.rectangle(img_color, (left, top), (left + width, top + height), (0, 0, 255), 5)  # 검출한 영역에 사각형과 원 표시
        cv2.circle(img_color, (center_x, center_y), 10, (0, 0, 255), -1)

    if max_index_b != -1:
        center_x = int(centroids_b[max_index_b, 0])
        center_y = int(centroids_b[max_index_b, 1])
        left = stats_b[max_index_b, cv2.CC_STAT_LEFT]
        top = stats_b[max_index_b, cv2.CC_STAT_TOP]
        width = stats_b[max_index_b, cv2.CC_STAT_WIDTH]
        height = stats_b[max_index_b, cv2.CC_STAT_HEIGHT]

        cv2.rectangle(img_color, (left, top), (left + width, top + height), (255, 0, 0), 5)  # 검출한 영역에 사각형과 원 표시
        cv2.circle(img_color, (center_x, center_y), 10, (255, 0, 0), -1)

    # cv.imshow('Red', img_mask_r)
    # cv.imshow('Blue', img_mask_b)
    cv2.imshow('Result', img_color)
    if not retval:
        break
    try:
    ## 조감도 wrapped img
        wrapped_img, minverse = wrapping(dst)
        #cv2.imshow('wrapped', wrapped_img)

    ## 조감도 필터링
        w_f_img = color_filter(wrapped_img)
        cv2.imshow('w_f_img', w_f_img)

    ##조감도 필터링 자르기
        w_f_r_img = roi(w_f_img)
    # cv2.imshow('w_f_r_img', w_f_r_img)

    ## 조감도 선 따기 wrapped img threshold
        _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(_gray, 160, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', thresh)

    ## 선 분포도 조사 histogram
        leftbase, rightbase = plothistogram(thresh)
    # plt.plot(hist)
    # plt.show()

    ## histogram 기반 window roi 영역
        draw_info = slide_window_search(thresh, leftbase, rightbase)
    # plt.plot(left_fit)
    # plt.show()

    ## 원본 이미지에 라인 넣기
        meanPts, result = draw_lane_lines(dst, thresh, minverse, draw_info)
        #cv2.imshow("result", result)
        center_point = int(meanPts[0][300][0])
        cv2.circle(result,(center_point+20,300),20,(0,255,0),-1)###120tae
        cv2.circle(result,(424,300),10,(255,0,0),-1)  ###tae
        cv2.imshow('result',result)
        control = int((center_point-424)/10+5)
        print('control:',control) ###컨트롤값 전송 0~8
        if i % 5 == 0:
            if (control == 0):
                ser.write(b'0')
            if (control == 1):
                ser.write(b'1')
            if (control == 2):
                ser.write(b'2')
            if (control == 3):
                ser.write(b'3')
            if (control == 4):
                ser.write(b'4')
            if (control == 5):
                ser.write(b'5') #직진
            if (control == 6):
                ser.write(b'6')
            if (control == 7):
                ser.write(b'7')
            if (control == 8):
                ser.write(b'8')


    except:
        continue
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break




cv2.waitKey(0)
