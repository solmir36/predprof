import time
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

GPIO.output(IN3, GPIO.HIGH)
GPIO.output(IN4, GPIO.LOW)
mr = GPIO.PWM(ENB, 1000)
mr.stop()

ml.stop()
mr.stop()
