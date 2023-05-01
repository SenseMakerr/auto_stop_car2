#!/usr/bin/env python

import roslib
import rospy
from std_msgs.msg import String
import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define motor driver pins
left_motor_in1 = 11
left_motor_in2 = 13
right_motor_in1 = 10
right_motor_in2 = 12

# Set up motor driver pins
GPIO.setup(left_motor_in1, GPIO.OUT)
GPIO.setup(left_motor_in2, GPIO.OUT)
GPIO.setup(right_motor_in1, GPIO.OUT)
GPIO.setup(right_motor_in2, GPIO.OUT)


def control_motor(command):
    if command == "forward":
        GPIO.output(left_motor_in1, True)
        GPIO.output(left_motor_in2, False)
        GPIO.output(right_motor_in1, True)
        GPIO.output(right_motor_in2, False)
    elif command == "stop":
        GPIO.output(left_motor_in1, False)
        GPIO.output(left_motor_in2, False)
        GPIO.output(right_motor_in1, False)
        GPIO.output(right_motor_in2, False)


def stop_sign_status_callback(msg):
    control_motor(msg.data)


def main():
    rospy.init_node('motor_controller', anonymous=True)
    rospy.Subscriber('/stop_sign_status', String, stop_sign_status_callback)
    rospy.spin()

if __name__ == '__main__':
    main()



