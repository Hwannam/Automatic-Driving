import cv2 as cv

cap = cv.VideoCapture(0)

while True:

    ret, img_color = cap.read()

    img_color = cv.flip(img_color, 1)

    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    hue_red = 0 # Red에 대한 hue값 설정
    lower_red = (hue_red - 10, 150, 0)
    upper_red = (hue_red + 10, 255, 255)

    hue_blue = 120 # Blue에 대한 hue값 설정
    lower_blue = (hue_blue - 20, 150, 0)
    upper_blue = (hue_blue + 20, 255, 255)

    img_mask_r = cv.inRange(img_hsv, lower_red, upper_red)
    img_mask_b = cv.inRange(img_hsv,lower_blue, upper_blue)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_mask_r = cv.morphologyEx(img_mask_r, cv.MORPH_DILATE, kernel, iterations=3) # 모폴로지 팽창을 통한 비어있는 부분 채우기
    img_mask_b = cv.morphologyEx(img_mask_b, cv.MORPH_DILATE, kernel, iterations=3)

    nlabels_r, labels_r, stats_r, centroids_r = cv.connectedComponentsWithStats(img_mask_r)
    nlabels_b, labels_b, stats_b, centroids_b = cv.connectedComponentsWithStats(img_mask_b)

    max_r = -1
    max_b = -1
    max_index_r = -1
    max_index_b = -1

    for x in range(nlabels_r):

        if x < 1:
            continue

        area_red = stats_r[x, cv.CC_STAT_AREA]

        if area_red > 10000: # 10000이상 범위로 검출시 신호 전송
            max_r = area_red
            max_index_r = x
            print("빨강")

    for y in range(nlabels_b):

        if y < 1:
            continue

        area_blue = stats_b[y, cv.CC_STAT_AREA]

        if area_blue > 10000: # 10000이상 범위로 검출시 신호 전송
            max_b = area_blue
            max_index_b = y
            print("파랑")

    if max_index_r != -1:
        center_x = int(centroids_r[max_index_r, 0])
        center_y = int(centroids_r[max_index_r, 1])
        left = stats_r[max_index_r, cv.CC_STAT_LEFT]
        top = stats_r[max_index_r, cv.CC_STAT_TOP]
        width = stats_r[max_index_r, cv.CC_STAT_WIDTH]
        height = stats_r[max_index_r, cv.CC_STAT_HEIGHT]

        cv.rectangle(img_color, (left, top), (left + width, top + height), (0, 0, 255), 5) # 검출한 영역에 사각형과 원 표시
        cv.circle(img_color, (center_x, center_y), 10, (0, 0, 255), -1)

    if max_index_b != -1:
        center_x = int(centroids_b[max_index_b, 0])
        center_y = int(centroids_b[max_index_b, 1])
        left = stats_b[max_index_b, cv.CC_STAT_LEFT]
        top = stats_b[max_index_b, cv.CC_STAT_TOP]
        width = stats_b[max_index_b, cv.CC_STAT_WIDTH]
        height = stats_b[max_index_b, cv.CC_STAT_HEIGHT]

        cv.rectangle(img_color, (left, top), (left + width, top + height), (255, 0, 0), 5) # 검출한 영역에 사각형과 원 표시
        cv.circle(img_color, (center_x, center_y), 10, (255, 0, 0), -1)

    cv.imshow('Red', img_mask_r)
    cv.imshow('Blue', img_mask_b)
    cv.imshow('Result', img_color)

    key = cv.waitKey(1)
    if key == 27:  # esc
        break
cv.destroyWindow()
