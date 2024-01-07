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

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, input_features)

    def forward(self, x, edge_index):
        # 첫 번째 GCN 레이어
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 두 번째 GCN 레이어
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 세 번째 GCN 레이어
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return x

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.color_sub = message_filters.Subscriber('/camera/color/image_rect_color', Image)
        self.info_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.pcd_pub = rospy.Publisher("/open3d_pointcloud", PointCloud2, queue_size=10)

        self.rate = rospy.Rate(30)

        # 동기화된 메시지 필터
        self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.info_sub], 10, 0.5)
        self.ts.registerCallback(self.callback)

        # VoxelBlockGrid 초기화
        self.device = o3c.Device("CUDA:0")  # 'CUDA:0' 또는 'CPU:0'로 설정하세요
        self.voxel_size = 2.0 / 512
        self.block_resolution = 8
        self.block_count = 50000
        self.depth_scale = 1000.0
        self.depth_max = 1.0
        self.vbg = None

        self.frame_count = 5

        # 깊이와 컬러 이미지를 저장할 큐 초기화
        self.depth_queue = deque(maxlen=self.frame_count)
        self.color_queue = deque(maxlen=self.frame_count)

        self.gcn = GCN(input_features=6, hidden_channels=64)

    def callback(self, depth_msg, color_msg, info_msg):
        try:
            # 이미지 변환
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")

            # 큐에 이미지 추가
            self.depth_queue.append(depth_image)
            self.color_queue.append(color_image)

            # 카메라 내부 파라미터 추출
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(info_msg.width, info_msg.height, info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5])
            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix,
                                        o3d.core.Dtype.Float64)
            # 3D 재구성 수행
            self.integrate(intrinsic, self.depth_scale, self.depth_max, info_msg)

        except CvBridgeError as e:
            print(e)

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
        self.pcd_pub.publish(cloud_data)

    def integrate(self, intrinsic, depth_scale, depth_max, camera_info):
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=self.device
        )

        for depth_image, color_image in zip(self.depth_queue, self.color_queue):
            start = time.time()

            # OpenCV 이미지를 Open3D 이미지로 n변환
            depth_o3d = o3d.t.geometry.Image(depth_image).to(self.device)
            color_o3d = o3d.t.geometry.Image(color_image).to(self.device)

            # 카메라 외부 파라미터 (여기서는 단순화를 위해 단위 행렬 사용)
            # 실제 응용에서는 extrinsic_id를 사용하여 적절한 변환 행렬을 설정해야 합니다.
            extrinsic = np.eye(4)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)

            frustum_block_coords = self.vbg.compute_unique_block_coordinates(
                    depth_o3d, intrinsic, extrinsic, depth_scale, depth_max)

            self.vbg.integrate(frustum_block_coords, depth_o3d, color_o3d, intrinsic,
                               intrinsic, extrinsic, depth_scale, depth_max)

            dt = time.time() - start
            # print('Finished integrating frames in {} seconds'.format(dt))

        pcd = self.vbg.extract_point_cloud().to_legacy()
        mesh = self.vbg.extract_triangle_mesh().to_legacy()

        if not mesh.is_empty():
            o3d.visualization.draw([mesh])

            graph_data = self.mesh_to_graph(mesh)
            gcn_output = self.gcn(graph_data.x, graph_data.edge_index).to("cpu").detach().numpy().copy()

            vertices = gcn_output[:,:3]
            colors = gcn_output[:,3:]

            pcd = o3d.t.geometry.PointCloud(vertices)
            pcd.point.colors = colors
            pcd = pcd.to_legacy()

        if not pcd.is_empty():
            self.publish_pointcloud(pcd, camera_info)

    def mesh_to_graph(self, mesh):
        # 메시에서 그래프 데이터 생성
        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)  # 정점의 색상 정보

        # 엣지 인덱스 생성
        edges = set()
        for triangle in np.asarray(mesh.triangles):
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((triangle[i], triangle[j]))

        if len(edges) == 0:
            raise ValueError("엣지 인덱스가 비어있습니다.")

        features = np.concatenate([vertices, colors], axis=1)
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        print(x.shape)
        print(np.asarray(list(edges)).shape)

        return Data(x=x, edge_index=edge_index)

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
