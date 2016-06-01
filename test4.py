import numpy as np
import cv2

# img = cv2.imread('test.png')

cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()

    r, h, c, w = 250, 90, 400, 125
    track_window = (c, r, w, h)

    roi = img[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(img, [pts], True, 255, 2)

    cv2.imshow('img2', img2)

    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()
