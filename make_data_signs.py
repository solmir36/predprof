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

minh = 137
mins = 45
minv = 27
maxh = 255
maxs = 255
maxv = 255

i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (minh, mins, minv), (maxh, maxs, maxv))
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours):
        rect = cv.boundingRect(max(contours, key=cv.contourArea))
        sign = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        cv.imwrite("imgs/" + str(i) + ".jpg", sign)
        cv.imshow("1", sign)

    i += 1
    rawCapture.truncate(0)
    
    if cv.waitKey(1) == 27:
        break
