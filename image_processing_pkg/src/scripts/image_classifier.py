#!/usr/bin/env python
import numpy as np
from operator import itemgetter
import roslib
import rospy

from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def load_trained_model(model_path):
    model = models.vgg16(pretrained=False)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(img_tensor, model):
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = nn.functional.softmax(outputs, dim=1).squeeze(0)
    return probabilities


def image_callback(msg):
    try:
        # Convert ROS image to OpenCV image
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        # Preprocess the image for classifier model
        image_tensor = preprocess(cv_image)
        # Classify the image
        model_path = "./model/finetuned_vgg16.pth"
        model = load_trained_model(model_path)
        model = model.to(device)

        pred = predict(image_tensor, model)
        handle_result(pred)

    except CvBridgeError as e:
        print(e)


# Global publisher
status_publisher = None


def is_stop_sign(pred):
    stop_sign = False
    if pred[0] >= 0.5:
        stop_sign = True
    return stop_sign


def handle_result(pred):
    global status_publisher
    if is_stop_sign(pred):
        status_publisher.publish("stop")
    else:
        status_publisher.publish("forward")


def main():
    global status_publisher
    rospy.init_node('image_classifier', anonymous=True)
    status_publisher = rospy.Publisher('/stop_sign_status', String, queue_size = 10)
    rospy.Subscriber("/raspi_camera/image_raw", Image, image_callback)
    rospy.spin()


if __name__ == '__main__':
    main()