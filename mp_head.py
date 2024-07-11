import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import time
from typing import List

# from lib.options import BaseOptions as MyBaseOptions
# from scripts.mycam_mask import img2masks
# from scripts.mycam_strand import img2strand
# from scripts.img2depth import img2depth

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode


BG_COLOR = (192, 192, 192) # gray
BLACK_COLOR = (0, 0, 0) # black
MASK_COLOR = (255, 255, 255) # white
BODY_COLOR = (0, 255, 0) # green
FACE_COLOR = (255, 0, 0) # red
CLOTHES_COLOR = (255, 0, 255) # purple

# 0 - background
# 1 - hair
# 2 - body-skin
# 3 - face-skin
# 4 - clothes
# 5 - others (accessories)
class App:
    def __init__(self):
        self.output_image = None
        self.output_image_face = None
        self.output_image_human = None
        self.output_image_human_color = None

        # Create an FaceLandmarker object.
        self.base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite',
                                               delegate=python.BaseOptions.Delegate.GPU)

        # self.base_options = python.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite',
        #                                        delegate=python.BaseOptions.Delegate.CPU)

        # self.base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
        # self.base_options = python.BaseOptions(model_asset_path='deeplab_v3.tflite')

        self.options = ImageSegmenterOptions(base_options=self.base_options,
                                             running_mode=VisionRunningMode.LIVE_STREAM,
                                             output_category_mask=True,
                                             output_confidence_masks=False,
                                             result_callback=self.mp_callback)

        self.segmenter = ImageSegmenter.create_from_options(self.options)

        self.latest_time_ms = 0

    def update(self, frame):
        t_ms = int(time.time() * 1000)
        if t_ms <= self.latest_time_ms:
            print("no update")
            return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.segmenter.segment_async(mp_image, t_ms)
        self.latest_time_ms = t_ms

    def mp_callback(self, segmentation_result: List[mp.Image], rgb_image: mp.Image, timestamp_ms: int):
        category_mask = segmentation_result.category_mask
        # confidence_mask = segmentation_result.confidence_mask

        image_data = rgb_image.numpy_view()
        fg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        fg_image[:] = MASK_COLOR[0]
        bg_image = np.zeros(image_data.shape[:2], dtype=np.uint8)
        bg_image[:] = BLACK_COLOR[0]

        condition1 = category_mask.numpy_view() == 1 # hair
        # condition1 = (category_mask.numpy_view() == 1) | (category_mask.numpy_view() == 3)
        condition2 = category_mask.numpy_view() == 3
        condition3 = (category_mask.numpy_view() == 4) | (category_mask.numpy_view() == 2) | (category_mask.numpy_view() == 3)
        # condition3 = category_mask.numpy_view() != 0
        if np.sum(condition1) == 0:
            self.output_image = bg_image
        else:
            self.output_image = np.where(condition1, fg_image, bg_image)
        if np.sum(condition2) == 0:
            self.output_image_face = bg_image
        else:
            self.output_image_face = np.where(condition2, fg_image, bg_image)
        if np.sum(condition3) == 0:
            self.output_image_human = bg_image
            self.output_image_human_color = image_data[:,:,::-1]
        else:
            self.output_image_human = np.where(condition3, np.ones(image_data.shape[:2], dtype=np.uint8), bg_image)
            self.output_image_human_color = self.output_image_human[:,:,np.newaxis] * image_data[:,:,::-1]

class RosApp(App):
    def __init__(self):
        super(RosApp, self).__init__()
        rospy.init_node('mediapipe_ros_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("segmented_mask", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("segmented_depth", Image, queue_size=1) # 추가: 깊이 정보 발행을 위한 퍼블리셔
        self.debug_pub = rospy.Publisher("segmented_image", Image, queue_size=1) # 추가: 깊이 정보 발행을 위한 퍼블리셔

        self.rate = rospy.Rate(30)
        self.cv_image = None
        self.cv_depth = None # 추가: 깊이 이미지 저장을 위한 변수

        self.header = None

        rospy.Subscriber("/camera/color/image_rect_color", Image, self.image_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback) # 추가: 깊이 이미지 구독

    def image_callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            # self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            rospy.logerr(e)

    def depth_callback(self, data): # 추가: 깊이 이미지 콜백 함수
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.header = data.header

        except CvBridgeError as e:
            rospy.logerr(e)

    def main(self):
        while not rospy.is_shutdown():
            if self.cv_image is not None and self.cv_depth is not None:
                self.update(self.cv_image)
                if self.output_image is not None:
                    try:
                        human_depth = np.where((self.output_image) & (self.cv_depth < 0.8*1000), self.cv_depth, 0)
                        debug_image = np.where(self.output_image[:,:,np.newaxis]>0, self.cv_image, 0)

                        ros_depth_image = self.bridge.cv2_to_imgmsg(human_depth, "16UC1")
                        ros_depth_image.header = self.header
                        self.depth_pub.publish(ros_depth_image)

                        ros_image = self.bridge.cv2_to_imgmsg(self.output_image, "8UC1")
                        ros_image.header = self.header
                        self.image_pub.publish(ros_image)

                        ros_debug_image = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                        ros_debug_image.header = self.header
                        self.debug_pub.publish(ros_debug_image)
                        self.output_image = None

                    except CvBridgeError as e:
                        rospy.logerr(e)

            self.rate.sleep()

if __name__ == "__main__":
    ros_app = RosApp()
    ros_app.main()
