import numpy as np
import cv2 as cv
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import tensorflow as tf
import line

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

model = tf.keras.models.load_model('model.h5')

def decode(one_hot_encode):
    label = ""
    if one_hot_encode == [1, 0, 0, 0, 0]:
        label = "speed20"
    elif one_hot_encode == [0, 1, 0, 0, 0]:
        label = "speed40"
    elif one_hot_encode == [0, 0, 1, 0, 0]:
        label = "hill"
    elif one_hot_encode == [0, 0, 0, 1, 0]:
        label = "stop"
    else:
        label = "none"
    return label

minh = 137
mins = 45
minv = 27
maxh = 255
maxs = 255
maxv = 255

def get_sign(image):
    cv.imshow("1", image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (minh, mins, minv), (maxh, maxs, maxv))
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours):
        rect = cv.boundingRect(max(contours, key=cv.contourArea))
        sign = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        cv.imshow("2", sign)

        im = cv.cvtColor(sign, cv.COLOR_BGR2GRAY)
        im = cv.resize(im, (64, 64))
        im = np.reshape(im, (im.shape[0], im.shape[1], 1))
        im = tf.cast(im, tf.float64) / 127.5 - 1.0
        im = tf.convert_to_tensor([im])

        pred = model.predict(im)[0]
        sign = decode(list(map(lambda x: x == max(pred), pred)))
        return sign
    return "none"

robot = line.Robot(0.6, 0)
speed = 20
i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    sign = get_sign(image)
    print(sign, speed)

    if sign == "stop":
        #robot.ml.ChangeDutyCycle(0)
        #robot.mr.ChangeDutyCycle(0)
        #break
        pass
    elif sign == "speed20":
        speed = 10
    elif sign == "speed40":
        speed = 15
    robot.line(image, speed)

    i += 1
    rawCapture.truncate(0)
    
    if cv.waitKey(1) == 27:
        break

