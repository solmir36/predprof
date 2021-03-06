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

cv.namedWindow("1")

minh = 0
mins = 0
minv = 0
maxh = 255
maxs = 255
maxv = 255

def empty(n):
    pass

cv.createTrackbar("minh", "1", 0, 255, empty)
cv.createTrackbar("mins", "1", 0, 255, empty)
cv.createTrackbar("minv", "1", 0, 255, empty)
cv.createTrackbar("maxh", "1", 0, 255, empty)
cv.createTrackbar("maxs", "1", 0, 255, empty)
cv.createTrackbar("maxv", "1", 0, 255, empty)

i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
	
    minh = cv.getTrackbarPos("minh", "1")
    mins = cv.getTrackbarPos("mins", "1")
    minv = cv.getTrackbarPos("minv", "1")
    maxh = cv.getTrackbarPos("maxh", "1")
    maxs = cv.getTrackbarPos("maxs", "1")
    maxv = cv.getTrackbarPos("maxv", "1")
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image, (minh, mins, minv), (maxh, maxs, maxv))
    cv.imshow("3", mask)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours):
        rect = cv.boundingRect(max(contours, key=cv.contourArea))
        cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
        sign = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        cv.imshow("2", sign)

    #cv.imwrite("imgs/" + str(i) + ".jpg", image)
    cv.imshow("1", image)

    i += 1
    rawCapture.truncate(0)
    
    if cv.waitKey(1) == 27:
        break
