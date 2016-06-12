import cv2
import numpy as np
import math
import copy

IMAGE_WIDTH = 36
IMAGE_HIGHT = 36
SIZE = 9
N_MIN_ACTIVE_PIXELS = 10
global EX_CHESSBOARD
EX_CHESSBOARD = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# sort the corners to remap the image


def getOuterPoints(rcCorners):
    ar = []
    ar.append(rcCorners[0, 0, :])
    ar.append(rcCorners[1, 0, :])
    ar.append(rcCorners[2, 0, :])
    ar.append(rcCorners[3, 0, :])

    x_sum = sum(rcCorners[x, 0, 0]
                for x in range(len(rcCorners))) / len(rcCorners)
    y_sum = sum(rcCorners[x, 0, 1]
                for x in range(len(rcCorners))) / len(rcCorners)

    def algo(v):
        return (math.atan2(v[0] - x_sum, v[1] - y_sum)
                + 2 * math.pi) % 2 * math.pi
        ar.sort(key=algo)
    return (ar[3], ar[0], ar[1], ar[2])

# examinate color
# red->1
# blue->-1


def checkColor(blur):
    r = 3 * IMAGE_WIDTH / 2
    chessboard = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            p = blur[(2 * i + 1) * r][(2 * j + 1) * r]
            if (p[0] < 100):
                chessboard[i][j] = 1
            elif (p[2] < 100):
                chessboard[i][j] = -1
            else:
                chessboard[i][j] = 0

    return chessboard


# classify chessmen


def checkChessmen(frame, EX_CHESSBOARD):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    cv2.imshow('blur',blur)
    # CHESSBOARD = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    CHESSBOARD = checkColor(blur)

    temp_sum = 0
    for i in range(len(EX_CHESSBOARD)):
        for j in range(len(EX_CHESSBOARD)):
            temp_sum += (CHESSBOARD[i][j] - EX_CHESSBOARD[i][j])
    # print temp_sum
    # print EX_CHESSBOARD
    # print CHESSBOARD
    # print
    if (abs(temp_sum) == 1):
        for i in range(len(EX_CHESSBOARD)):
            for j in range(len(EX_CHESSBOARD)):
                EX_CHESSBOARD[i][j] = CHESSBOARD[i][j]

        print EX_CHESSBOARD
        print


cap = cv2.VideoCapture(0)
big_rectangle = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]])
while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    cv2.imshow('frame',frame)
    # adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 15)
    # show
    # cv2.imshow('thresh',thresh)

    # find the countours
    # if you're using 2.4, you should delete img
    img, contours0, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # size of the image
    h, w = frame.shape[: 2]
    # copy the frame to  shwo the possible candidate
    candidates = frame.copy()
    # biggest rectagle
    size_rectangle_max = 0

    for i in range(len(contours0)):
        # approximate contours to polygons
        approximation = cv2.approxPolyDP(contours0[i], 4, True)
        # has the polygon 4 sides?
        if (not (len(approximation) == 4)):
            continue
        # is the polygon convex?
        if(not cv2.isContourConvex(approximation)):
            continue
        # area of the polygon
        size_rectangle = cv2.contourArea(approximation)
        # store the biggest
        if size_rectangle > size_rectangle_max:
            size_rectangle_max = size_rectangle
            big_rectangle = approximation
            # print approximation
    # show the best candidate
    approximation = big_rectangle
    for i in range(len(approximation)):
        cv2.line(candidates, (big_rectangle[(i % 4)][0][0], big_rectangle[(i % 4)][0][
            1]), (big_rectangle[((i + 1) % 4)][0][0], big_rectangle[((i + 1) %
                                                                     4)][0][1]), (255, 0, 0), 2)

    # cv2.imshow('candidates', candidates)

    # point to remap
    points1 = np.array([np.array([0.0, 0.0], np.float32) + np.array([324, 0], np.float32), np.array([0.0, 0.0], np.float32),
                        np.array([0.0, 0.0], np.float32) + np.array([0.0, 324], np.float32), np.array([0.0, 0.0], np.float32) + np.array([324, 324], np.float32), ], np.float32)

    outerPoints = getOuterPoints(big_rectangle)
    points2 = np.array(outerPoints, np.float32)
    # transfermation matrix
    pers = cv2.getPerspectiveTransform(points2, points1)
    # remap the image
    warp = cv2.warpPerspective(
        frame, pers, (SIZE * IMAGE_HIGHT, SIZE * IMAGE_WIDTH))
    # warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    # check color
    checkChessmen(warp, EX_CHESSBOARD)
    # show
    # cv2.imshow('test', warp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
