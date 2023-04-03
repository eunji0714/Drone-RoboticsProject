from utils import *
import cv2, keyboard, time
if __name__ == "__main__":
    tello = initTello()
    tello.streamon()

    while True:
        frame_read = tello.get_frame_read()

        img = frame_read.frame
        img = cv2.resize(img, (360, 240))
        cv2.imshow("drone",img)

        keyboard = cv2.waitKey(1)
        if keyboard & 0xFF == ord('q'):
            frame_read.stop()
            tello.steamoff()
