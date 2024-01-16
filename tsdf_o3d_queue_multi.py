#!/usr/bin/env python
import time
import rospy
import open3d as o3d
import open3d.core as o3c
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from collections import deque

import tf2_ros
import tf2_geometry_msgs

import tf

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth1_sub = message_filters.Subscriber('/masked_depth_image/camera1', Image)
        self.color1_sub = message_filters.Subscriber('/segmented_image/camera1', Image)
        self.depth2_sub = message_filters.Subscriber('/masked_depth_image/camera2', Image)
        self.color2_sub = message_filters.Subscriber('/segmented_image/camera2', Image)

        self.info1_sub = message_filters.Subscriber('/camera1/aligned_depth_to_color/camera_info', CameraInfo)
        self.info2_sub = message_filters.Subscriber('/camera2/aligned_depth_to_color/camera_info', CameraInfo)

        self.pub = rospy.Publisher("/open3d_pointcloud", PointCloud2, queue_size=10)
        self.rate = rospy.Rate(30)

        # 동기화된 메시지 필터
        self.ts = message_filters.ApproximateTimeSynchronizer([self.depth1_sub, self.color1_sub, self.info1_sub, self.depth2_sub, self.color2_sub, self.info2_sub], 10, 0.5)

        self.ts.registerCallback(self.callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # VoxelBlockGrid 초기화
        self.device = o3c.Device("CUDA:0")  # 'CUDA:0' 또는 'CPU:0'로 설정하세요
        self.voxel_size = 1.0 / 512
        self.block_resolution = 8
        self.block_count = 10000
        self.depth_scale = 1000.0
        self.depth_max = 2.0
        self.vbg = None

        self.frame_count = 10

        self.camera_main_frame_id = None

        # 깊이와 컬러 이미지를 저장할 큐 초기화
        self.depth_queue = deque(maxlen=self.frame_count)
        self.color_queue = deque(maxlen=self.frame_count)

    def get_extrinsic_matrix(self, target_frame, source_frame="base"):
        try:
            # ROS에서 TF 정보를 얻음
            trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0), rospy.Duration(1.0))
            t = trans.transform.translation
            r = trans.transform.rotation

            # Translation과 Rotation을 numpy 행렬로 변환
            translation = np.array([t.x, t.y, t.z])
            rotation = tf.transformations.quaternion_matrix([r.x, r.y, r.z, r.w])

            # Extrinsic 행렬 생성
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotation[:3, :3]
            extrinsic[:3, 3] = translation

            return extrinsic

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("TF2 Exception: %s" % str(e))
            return None

    def callback(self, depth1_msg, color1_msg, info1_msg, depth2_msg, color2_msg, info2_msg):
        try:
            # 두 카메라의 이미지 처리
            depth1_image = self.bridge.imgmsg_to_cv2(depth1_msg, "16UC1")
            color1_image = self.bridge.imgmsg_to_cv2(color1_msg, "rgb8")
            depth2_image = self.bridge.imgmsg_to_cv2(depth2_msg, "16UC1")
            color2_image = self.bridge.imgmsg_to_cv2(color2_msg, "rgb8")
            if not depth1_image.any() or not depth2_image.any():
                print("no depth image")
                return

            # extrinsic1 = self.get_extrinsic_matrix("base","camera1_link")
            extrinsic2 = self.get_extrinsic_matrix("camera1_color_optical_frame","camera2_color_optical_frame")
            extrinsic1 = np.eye(4)
            #extrinsic2 = np.eye(4)

            if not self.camera_main_frame_id:
                self.camera_main_frame_id = info1_msg.header.frame_id

            if extrinsic1 is not None and extrinsic2 is not None:
                self.depth_queue.append((depth1_image, info1_msg, extrinsic1))
                self.depth_queue.append((depth2_image, info2_msg, extrinsic2))

                self.color_queue.append(color1_image)
                self.color_queue.append(color2_image)

                # 3D 재구성 수행
                self.integrate()

        except Exception as e:
            print(e)

    def publish_pointcloud(self, pcd, camera_main_frame_id):
        # Open3D 포인트 클라우드를 numpy 배열로 변환
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        r, g, b = (colors * 255).astype(np.uint8).T

        rgba = np.left_shift(np.ones_like(r, dtype=np.uint32) * 255, 24) | \
            np.left_shift(r.astype(np.uint32), 16) | \
            np.left_shift(g.astype(np.uint32), 8) | \
            b.astype(np.uint32)

        points = np.concatenate((points, rgba[:, np.newaxis].astype(np.uint32)), axis=1, dtype=object)

        # ROS PointCloud2 메시지 생성
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = camera_main_frame_id

        # PointField 구조 정의
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('rgba', 12, pc2.PointField.UINT32, 1)]

        cloud_data = pc2.create_cloud(header, fields, points)

        # 포인트 클라우드 Publish
        self.pub.publish(cloud_data)

    def init_voxel_block_grid(self):
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=self.device
        )
        
    def integrate(self):
        # if not self.vbg:
        self.init_voxel_block_grid()

        for (depth_image, camera_info, extrinsic), color_image in zip(self.depth_queue, self.color_queue):
            start = time.time()

            # 이미지 변환 및 카메라 내부 파라미터 추출
            depth_o3d = o3d.t.geometry.Image(depth_image).to(self.device)
            color_o3d = o3d.t.geometry.Image(color_image).to(self.device)

            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(camera_info.width, camera_info.height, camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5])
            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)

            # extrinsic = np.eye(4)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)

            frustum_block_coords = self.vbg.compute_unique_block_coordinates(
                depth_o3d, intrinsic, extrinsic, self.depth_scale, self.depth_max)

            self.vbg.integrate(frustum_block_coords, depth_o3d, color_o3d, intrinsic,
                               intrinsic, extrinsic, self.depth_scale, self.depth_max)

            dt = time.time() - start
            # print('Frame integration took {} seconds'.format(dt)

        pcd = self.vbg.extract_point_cloud().to_legacy()
        if not pcd.is_empty():
            self.publish_pointcloud(pcd, self.camera_main_frame_id)

        self.vbg = None

def main():
    rospy.init_node('image_subscriber', anonymous=True)
    dis = ImageSubscriber()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        # 종료 작업
        # 예: dis.vbg.save('path_to_save.npz')

if __name__ == '__main__':
    main()
