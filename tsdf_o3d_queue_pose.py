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


class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_ns = rospy.get_param("camera_ns", "camera")
        # self.depth_sub = message_filters.Subscriber('/masked_depth_image/'+self.camera_ns, Image)
        # self.color_sub = message_filters.Subscriber('/segmented_image/'+self.camera_ns, Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.color_sub = message_filters.Subscriber('/camera/color/image_rect_color', Image)

        self.info_sub = message_filters.Subscriber('/'+self.camera_ns+'/aligned_depth_to_color/camera_info', CameraInfo)
        self.pub = rospy.Publisher("/open3d_pointcloud", PointCloud2, queue_size=10)
        self.rate = rospy.Rate(30)

        # 동기화된 메시지 필터
        self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.info_sub], 10, 0.5)
        self.ts.registerCallback(self.callback)

        # VoxelBlockGrid 초기화
        self.device = o3c.Device("CUDA:0")  # 'CUDA:0' 또는 'CPU:0'로 설정하세요
        self.voxel_size = 1.0 / 512
        self.block_resolution = 16
        self.block_count = 10000
        self.depth_scale = 1000.0
        self.depth_max = 0.8
        self.vbg = None

        self.frame_count = 5

        # 깊이와 컬러 이미지를 저장할 큐 초기화
        # self.depth_queue = deque(maxlen=self.frame_count)
        # self.color_queue = deque(maxlen=self.frame_count)

        self.prev_pcd = None

        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=self.device
        )


    def callback(self, depth_msg, color_msg, info_msg):
        # try:
        # 이미지 변환
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        self.color_image = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")

        # 큐에 이미지 추가
        if not self.depth_image.any():
            print("no depth image")
            return
        # self.depth_queue.append(depth_image)
        # self.color_queue.append(color_image)

        # 카메라 내부 파라미터 추출
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(info_msg.width, info_msg.height, info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5])
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                                    o3d.core.Dtype.Float64)
        # 3D 재구성 수행
        self.integrate(intrinsic, self.depth_scale, self.depth_max, info_msg)

        # except Exception as e:
        #     print(e)


    def publish_pointcloud(self, pcd, camera_info):
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
        header.frame_id = camera_info.header.frame_id

        # PointField 구조 정의
        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('rgba', 12, pc2.PointField.UINT32, 1)]

        cloud_data = pc2.create_cloud(header, fields, points)

        # 포인트 클라우드 Publish
        self.pub.publish(cloud_data)

    def integrate(self, intrinsic, depth_scale, depth_max, camera_info):
        # OpenCV 이미지를 Open3D 이미지로 변환
        depth_o3d = o3d.t.geometry.Image(self.depth_image).to(self.device)
        color_o3d = o3d.t.geometry.Image(self.color_image).to(self.device)


        # 현재 프레임의 포인트 클라우드 생성
        current_pcd = self.create_point_cloud(depth_o3d, color_o3d, intrinsic, np.eye(4))

        # ICP 변환 계산
        if self.prev_pcd is not None:
            icp_result = o3d.t.pipelines.registration.icp(
                current_pcd, self.prev_pcd, 0.02, np.identity(4),
                o3d.t.pipelines.registration.TransformationEstimationPointToPoint())

            # 외부 파라미터 설정
            extrinsic = icp_result.transformation
        else:
            extrinsic = np.eye(4)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)

        print(extrinsic)
        # VoxelBlockGrid에 프레임 통합
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth_o3d, intrinsic, extrinsic, depth_scale, depth_max)

        self.vbg.integrate(frustum_block_coords, depth_o3d, color_o3d, intrinsic,
                           intrinsic, extrinsic, depth_scale, depth_max)


        # 포인트 클라우드 추출 및 발행
        pcd = self.vbg.extract_point_cloud()
        print(pcd)

        if not pcd.is_empty():
            self.prev_pcd = pcd
            self.publish_pointcloud(pcd.to_legacy(), camera_info)

    def create_point_cloud(self, depth, color, intrinsic, extrinsic):
        rgbd = o3d.t.geometry.RGBDImage(color, depth, True)
        # 포인트 클라우드 다운샘플링
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic, self.depth_scale, self.depth_max)
        voxel_size = 0.01  # 보켈 크기 설정 (값은 상황에 맞게 조절)
        if not pcd.is_empty():
            pcd = pcd.voxel_down_sample(voxel_size)
        return pcd

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
