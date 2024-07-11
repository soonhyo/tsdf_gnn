import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('hair_point_cloud_processor', anonymous=True)
        self.point_cloud_sub = rospy.Subscriber('/open3d_pointcloud', PointCloud2, self.point_cloud_callback)
        self.ellipsoid_marker_sub = rospy.Subscriber('/ellipsoid_marker', Marker, self.ellipsoid_marker_callback)
        self.filled_cloud_pub = rospy.Publisher('/filled_point_cloud', PointCloud2, queue_size=10)
        self.point_cloud = None
        self.ellipsoid_marker = None

    def point_cloud_callback(self, msg):
        # Convert PointCloud2 message to Open3D PointCloud
        self.point_cloud = self.pointcloud2_to_open3d(msg)
        print(self.point_cloud)
        self.fill_point_cloud()

    def ellipsoid_marker_callback(self, msg):
        # Save the ellipsoid marker
        self.ellipsoid_marker = msg

    def pointcloud2_to_open3d(self, cloud_msg):
        # Convert sensor_msgs/PointCloud2 to Open3D PointCloud
        points = np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))))
        pcl_data = o3d.geometry.PointCloud()
        pcl_data.points = o3d.utility.Vector3dVector(points)
        return pcl_data

    def fill_point_cloud(self):
        if self.point_cloud is None or self.ellipsoid_marker is None:
            return

        # Extract ellipsoid parameters from the marker
        center = np.array([self.ellipsoid_marker.pose.position.x,
                           self.ellipsoid_marker.pose.position.y,
                           self.ellipsoid_marker.pose.position.z])

        scale = np.array([self.ellipsoid_marker.scale.x / 2,
                          self.ellipsoid_marker.scale.y / 2,
                          self.ellipsoid_marker.scale.z / 2])

        # Create an ellipsoid mesh
        ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        ellipsoid.scale(scale[0], [1, 0, 0])
        ellipsoid.scale(scale[1], [0, 1, 0])
        ellipsoid.scale(scale[2], [0, 0, 1])
        ellipsoid.translate(center)
        ellipsoid.compute_vertex_normals()

        # Simplify the ellipsoid mesh
        voxel_size = 0.001  # Adjust this value to control the level of simplification
        simplified_ellipsoid = ellipsoid.simplify_vertex_clustering(voxel_size)

        ellipsoid_points = np.asarray(simplified_ellipsoid.vertices)
        ellipsoid_normals = np.asarray(simplified_ellipsoid.vertex_normals)

        # Create a filled point cloud by adding points between the ellipsoid and the point cloud
        filled_points = self.create_filled_point_cloud(ellipsoid_points, ellipsoid_normals, np.asarray(self.point_cloud.points), center, scale)
        print(ellipsoid_points.shape)
        # Convert to PointCloud2 and publish
        self.publish_filled_point_cloud(filled_points)

    def create_filled_point_cloud(self, ellipsoid_points, ellipsoid_normals, cloud_points, center, scale, num_points=100):
        filled_points = []

        for i in range(len(ellipsoid_points)):
            ellipsoid_point = ellipsoid_points[i]
            normal = ellipsoid_normals[i]
            direction = normal / np.linalg.norm(normal)
            print("c1")
            # Find the intersection point with the point cloud in the direction of the normal
            intersections = self.find_intersections(cloud_points, ellipsoid_point, direction)
            if intersections is None:
                continue

            cloud_point = intersections[0]  # Using the first intersection point
            print("c2")
            # Interpolate points between the ellipsoid point and the cloud point
            for j in range(num_points):
                t = j / float(num_points)
                point = (1 - t) * ellipsoid_point + t * cloud_point
                filled_points.append(point)
                print("c3")
        return np.array(filled_points)

    def find_intersections(self, cloud_points, ellipsoid_point, direction, tolerance=0.1):
        intersections = []

        for cloud_point in cloud_points:
            vector = cloud_point - ellipsoid_point
            distance = np.dot(vector, direction)
            if abs(distance) < tolerance:
                intersections.append(cloud_point)

        return intersections if intersections else None

    def publish_filled_point_cloud(self, filled_points):
        # Create PointCloud2 message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_color_optical_frame"

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        print("d")
        cloud_data = filled_points.astype(np.float32)

        cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        self.filled_cloud_pub.publish(cloud_msg)

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
