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

mask_min = [200, 100, 100]
mask_max = [255, 255, 255]

def minh(v):
    mask_min[0] = v
def mins(v):
    mask_min[1] = v
def minv(v):
    mask_min[2] = v
def maxh(v):
    mask_max[0] = v
def maxs(v):
    mask_max[1] = v
def maxv(v):
    mask_max[2] = v

cv.createTrackbar("minh", "1", 0, 255, minh)
cv.createTrackbar("mins", "1", 0, 255, mins)
cv.createTrackbar("minv", "1", 0, 255, minv)
cv.createTrackbar("maxh", "1", 0, 255, maxh)
cv.createTrackbar("maxs", "1", 0, 255, maxs)
cv.createTrackbar("maxv", "1", 0, 255, maxv)

i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    #hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image, mask_min, mask_max)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours):
        rect = cv.boundingRect(max(contours, key=cv.contourArea))
        cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))

    #cv.imwrite("imgs/" + str(i) + ".jpg", image)
    cv.imshow("1", image)

    i += 1
    rawCapture.truncate(0)
    
    if cv.waitKey(1) == 27:
        break
