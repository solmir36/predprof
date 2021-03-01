import numpy as np
import cv2 as cv
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    cv.imwrite("imgs/" + str(i) + ".jpg", image)
    cv.imshow("1", image)

    i += 1
    rawCapture.truncate(0)
    
    if cv.waitKey(1) == 27:
        break
