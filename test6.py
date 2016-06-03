import readchess
import cv2


chess = readchess.ReadChess()
while (True):

    candidates, warp, chessboard = chess.getChess()
    cv2.imshow('candidate', candidates)
    cv2.imshow('warp', warp)
    print chessboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
