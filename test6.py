import readmorechess
import cv2
import numpy as np

chess = readmorechess.ReadChess()
while (True):

    candidates, warp, chessboard = chess.getChess()
    cv2.imshow('candidate', candidates)
    cv2.imshow('warp', warp)
    chessboard=np.array(chessboard)
    print chessboard
    print
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()