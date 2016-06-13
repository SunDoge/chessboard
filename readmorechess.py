# computer vision
import cv2
import math
import numpy


class ReadChess(object):
    """docstring for ReadChess"""

    def __init__(self, chessboard=[[2, 2, 2, 2, 2, 2, 2, 2, 2],
                                   [2, 0, 2, 2, 0, 2, 2, 0, 2],
                                   [2, 2, 0, 2, 0, 2, 0, 2, 2],
                                   [2, 2, 2, 0, 0, 0, 2, 2, 2],
                                   [2, 0, 0, 0, 2, 0, 0, 0, 2],
                                   [2, 2, 2, 0, 0, 0, 2, 2, 2],
                                   [2, 2, 0, 2, 0, 2, 0, 2, 2],
                                   [2, 0, 2, 2, 0, 2, 2, 0, 2],
                                   [2, 2, 2, 2, 2, 2, 2, 2, 2]]):
        # super(ReadChess, self).__init__()
        self.__IMAGE_WIDTH = 48
        self.__IMAGE_HIGHT = 48
        self.__SIZE = 8
        self.__CHESSBOARD = chessboard
        if (not('cap' in dir())):
            self.cap = cv2.VideoCapture(0)

    # sort the corners to remap the image
    def getOuterPoints(self, rcCorners):
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
    def checkColor(self, blur):
        r = self.__IMAGE_HIGHT
        chessboard = self.__CHESSBOARD
        for i in range(1, 8):
            for j in range(1, 8):
                p = blur[i * r + 5][j * r + 5]

                if (chessboard[i][j] < 2):
                    # print p
                    if (p[0] < 100):
                        chessboard[i][j] = 1
                    elif (p[2] < 100):
                        chessboard[i][j] = -1
                    else:
                        chessboard[i][j] = 0
                # cv2.circle(blur, (i * r, j * r), 5, (0, 0, 255), -1)
        return chessboard

    # classify chessmen
    def checkChessmen(self, frame):
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # CHESSBOARD = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        CHESSBOARD = self.checkColor(blur)
        # cv2.imshow('blur',blur)
        # temp_sum = 0
        # for i in range(len(self.EX_CHESSBOARD)):
        #     for j in range(len(self.EX_CHESSBOARD)):
        #         temp_sum += (CHESSBOARD[i][j] - self.EX_CHESSBOARD[i][j])

        # if (abs(temp_sum) == 1):
        #     # for i in range(len(self.EX_CHESSBOARD)):
        #     #     for j in range(len(self.EX_CHESSBOARD)):
        #     #         EX_CHESSBOARD[i][j] = CHESSBOARD[i][j]
        #     self.EX_CHESSBOARD = CHESSBOARD
        #     print self.EX_CHESSBOARD
        #     print

    # print CHESSBOARD
    def getChess(self):

        big_rectangle = numpy.array(
            [[[0, 0]], [[0, 48]], [[48, 48]], [[48, 0]]])
        ret, frame = self.cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 15)
        # show
        # cv2.imshow('thresh',thresh)

        # find the countours
        # if you're using 2.4, you should delete img
        img, contours0, hierarchy = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # size of the image
        h, w = frame.shape[:2]
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
        points1 = numpy.array([numpy.array([0.0, 0.0], numpy.float32) + numpy.array([384, 0], numpy.float32), numpy.array([0.0, 0.0], numpy.float32),
                               numpy.array([0.0, 0.0], numpy.float32) + numpy.array([0.0, 384], numpy.float32), numpy.array([0.0, 0.0], numpy.float32) + numpy.array([384, 384], numpy.float32), ], numpy.float32)

        outerPoints = self.getOuterPoints(big_rectangle)
        points2 = numpy.array(outerPoints, numpy.float32)
        # transfermation matrix
        pers = cv2.getPerspectiveTransform(points2, points1)
        # remap the image
        warp = cv2.warpPerspective(
            frame, pers, (self.__SIZE * self.__IMAGE_HIGHT, self.__SIZE * self.__IMAGE_WIDTH))
        # warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        # check color
        self.checkChessmen(warp)
        # cv2.imshow('chessboard', warp)
        return candidates, warp, self.__CHESSBOARD
