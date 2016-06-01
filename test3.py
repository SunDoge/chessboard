import cv2
import numpy as np

img = cv2.imread('test.png')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)

print dst
dst = cv2.dilate(dst, None)
ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(
    centroids), (5, 5), (-1, -1), criteria)
print corners


img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('test', dst)
cv2.imshow('te', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
