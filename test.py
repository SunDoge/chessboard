#!/usr/bin/env python
# -*- coding: utf-8 -*-
'a test module'
__author__ = 'SunDoge'
import numpy as np
import cv2
#import glob
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
lower_red = np.array([-10, 50, 50])
upper_red = np.array([10, 255, 255])
black = np.array([0, 0, 0])
objp = np.zeros((6 * 7, 3), np.float32)
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

while (True):

    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)

    ret, corners = cv2.findChessboardCorners(gray, (3, 3), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    # dst = np.uint8(dst)

    # frame[dst > 0.01 * dst.max()] = [0, 0, 255]
    frame[imgpoints] = [0, 0, 255]

    player1 = cv2.inRange(hsv, lower_blue, upper_blue)
    player2 = cv2.inRange(hsv, lower_red, upper_red)
    cv2.imshow('frame', frame)
    cv2.imshow('player1', player1)
    cv2.imshow('player2', player2)
    #cv2.imshow('gray', gray)
    #cv2.imshow('dst', frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
