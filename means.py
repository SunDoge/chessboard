import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
while (1):

    img = [cap.read()[1] for i in xrange(5)]

    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

    gray = [np.float64(i) for i in gray]

    noise = np.random.randn(*gray[1].shape) * 10

    noisy = [i + noise for i in gray]

    noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]

    dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)

    plt.subplot(131), plt.imshow(gray[2], 'gray')
    plt.subplot(132), plt.imshow(noisy[2], 'gray')
    plt.subplot(133), plt.imshow(dst, 'gray')
    plt.show()

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
