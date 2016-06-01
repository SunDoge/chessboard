import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while (True):

	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('gray', gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.imwrite('test.png',gray)
		break

cv2.destroyAllWindows()