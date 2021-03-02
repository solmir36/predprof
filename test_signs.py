import numpy as np
import cv2 as cv
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import tensorflow as tf

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
model = tf.keras.models.load_model('model.h5')

def decode(one_hot_encode):
    label = ""
    if one_hot_encode == [1, 0, 0, 0, 0]:
        label = "none"
    elif one_hot_encode == [0, 1, 0, 0, 0]:
        label = "speed20"
    elif one_hot_encode == [0, 0, 1, 0, 0]:
        label = "speed40"
    elif one_hot_encode == [0, 0, 0, 1, 0]:
        label = "hill"
    else:
        label = "stop"
    return label

i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    #cv.imwrite("imgs/" + str(i) + ".jpg", image)
    #cv.imshow("1", image)

    im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    im = np.reshape(im, (im.shape[0], im.shape[1], 1))
    im = tf.cast(im, tf.float64) / 127.5 - 1.0
    im = tf.convert_to_tensor([im])

    pred = model.predict(im)[0]
    sign = decode(list(map(lambda x: x == max(pred), pred)))

    print(sign)

    i += 1
    rawCapture.truncate(0)
    
    if cv.waitKey(1) == 27:
        break
