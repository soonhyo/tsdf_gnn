#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import mediapipe as mp
import numpy as np

class ImageDepthProcessor:
    def __init__(self):
        self.node_name = "image_depth_processor"
        rospy.init_node(self.node_name)

        self.depth_pub = rospy.Publisher("/segmented_depth", Image, queue_size=10)
        self.image_pub = rospy.Publisher("/segmented_image", Image, queue_size=10)

        self.bridge = CvBridge()

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

        self.cv_image = None
        self.cv_depth = None
        self.header = None

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def depth_callback(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.header = data.header
        except CvBridgeError as e:
            rospy.logerr(e)

    def process_image(self):
        if self.cv_image is not None and self.cv_depth is not None:
            image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(image)
            if results.segmentation_mask is not None:
                # Create a binary mask
                mask = (results.segmentation_mask > 0.5)

                # Apply the mask to depth image
                segmented_depth = mask * self.cv_depth
                segmented_image = mask[:,:,np.newaxis] * self.cv_image

                try:
                    depth_msg = self.bridge.cv2_to_imgmsg(segmented_depth, "16UC1")
                    if self.header is not None:
                        depth_msg.header = self.header
                    self.depth_pub.publish(depth_msg)

                    image_msg = self.bridge.cv2_to_imgmsg(segmented_image, "rgb8")
                    if self.header is not None:
                        image_msg.header = self.header
                    self.image_pub.publish(image_msg)

                except CvBridgeError as e:
                    rospy.logerr(e)

    def run(self):
        rate = rospy.Rate(30)  # 10 Hz
        while not rospy.is_shutdown():
            self.process_image()
            rate.sleep()
if __name__ == '__main__':
    processor = ImageDepthProcessor()
    processor.run()
