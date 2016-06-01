#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(frame, 100, 200, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                            minLineLength, maxLineGap)
    # print lines
    for x1,y1,x2,y2 in lines[0]:
    	
    # cv2.imshow('blur', blur)

    cv2.imshow('edges', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows
cap.release()
