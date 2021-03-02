import numpy as np
import cv2 as cv
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import math
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

IN1 = 1
IN2 = 7
IN3 = 8
IN4 = 25
ENA = 12
ENB = 13

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
ml = GPIO.PWM(ENA, 1000)
ml.stop()
ml.start(100)

GPIO.output(IN3, GPIO.HIGH)
GPIO.output(IN4, GPIO.LOW)
mr = GPIO.PWM(ENB, 1000)
mr.stop()
mr.start(100)

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)

kp = 0.4
kd = 0
erld = 0

def detect_edges(frame):
    lower_blue = np.array([90, 120, 0], dtype = "uint8") # lower limit of blue color
    upper_blue = np.array([150, 255, 255], dtype="uint8") # upper limit of blue color
    mask = cv.inRange(hsv,lower_blue,upper_blue) # this mask will filter out everything but blue

    # detect edges
    edges = cv.Canny(mask, 50, 100) 
    return edges

def region_of_interest(edges):
    height, width = edges.shape # extract the height and width of the edges frame
    mask = np.zeros_like(edges) # make an empty matrix with same dimensions of the edges frame

    # only focus lower half of the screen
    # specify the coordinates of 4 points (lower left, upper left, upper right, lower right)
    polygon = np.array([[
        (0, height), 
        (0,  height/2),
        (width , height/2),
        (width , height),
    ]], np.int32)

    cv.fillPoly(mask, polygon, 255) # fill the polygon with blue color 
    cropped_edges = cv.bitwise_and(edges, mask) 
    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1  
    theta = np.pi / 180  
    min_threshold = 10 
    line_segments = cv.HoughLinesP(cropped_edges, rho, theta, min_threshold, 
                                    np.array([]), minLineLength=5, maxLineGap=0)
    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1/3

    left_region_boundary = width * (1 - boundary) 
    right_region_boundary = width * boundary 
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    # lane_lines is a 2-D array consisting the coordinates of the right and left lane lines
    # for example: lane_lines = [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    # where the left array is for left lane and the right array is for right lane 
    # all coordinate points are in pixels
    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0: 
        slope = 0.1    

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape
     
    if len(lane_lines) == 2: # if two lane lines are detected
        _, _, left_x2, _ = lane_lines[0][0] # extract left x2 from lane_lines array
        _, _, right_x2, _ = lane_lines[1][0] # extract right x2 from lane_lines array
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)  
    elif len(lane_lines) == 1: # if only one line is detected
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)
    
    elif len(lane_lines) == 0: # if no line is detected
        x_offset = 0
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
    steering_angle = angle_to_mid_deg + 90 

    return steering_angle

i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    edges = detect_edges(hsv)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(image, line_segments)
    steering_angle = get_steering_angle(image, lane_lines)

    err = steering_angle - 90
    up = err * kp + (err - erld) * kd
    erld = err
    vl = 30 + up
    vr = 30 - up
    
    if abs(vl) >= 100:
        vl = vl // abs(vl)
    if abs(vr) >= 100:
        vr = vr // abs(vr)

    if vl < 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        vl = -vl
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)

    if vr < 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        vr = -vr
    else:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    print(vl, " ", vr)

    ml.ChangeDutyCycle(vl);
    mr.ChangeDutyCycle(vr);
    
    i += 1
    rawCapture.truncate(0)
    
    if cv.waitKey(1) == 27:
        break

