#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def capture_and_publish():
    # Initialize the camera publisher and CvBridge
    pub = rospy.Publisher('/raspi_camera/image_raw', Image, queue_size=10)
    bridge = CvBridge()

    # Set up the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera index
    if not cap.isOpened():
        rospy.logerr("Unable to open camera")
        return

    # Capture and publish the images
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            try:
                ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
                pub.publish(ros_image)
            except CvBridgeError as e:
                rospy.logerr("CvBridge error: {}".format(e))
        rospy.sleep(0.01)

    cap.release()


def main():
    rospy.init_node('camera_publisher', anonymous=True)
    try:
        capture_and_publish()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

