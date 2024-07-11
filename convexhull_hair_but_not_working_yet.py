import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('point_cloud_processor', anonymous=True)
        self.point_cloud_sub = rospy.Subscriber('/open3d_pointcloud', PointCloud2, self.point_cloud_callback)
        self.ellipsoid_marker_sub = rospy.Subscriber('/ellipsoid_marker', Marker, self.ellipsoid_marker_callback)
        self.filled_cloud_pub = rospy.Publisher('/filled_point_cloud', PointCloud2, queue_size=10)
        self.point_cloud = None
        self.ellipsoid_marker = None

    def point_cloud_callback(self, msg):
        # Convert PointCloud2 message to Open3D PointCloud
        self.point_cloud = self.pointcloud2_to_open3d(msg)
        self.process_and_publish()

    def ellipsoid_marker_callback(self, msg):
        # Save the ellipsoid marker
        self.ellipsoid_marker = msg
        self.process_and_publish()

    def pointcloud2_to_open3d(self, cloud_msg):
        # Convert sensor_msgs/PointCloud2 to Open3D PointCloud
        points = np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))))
        pcl_data = o3d.geometry.PointCloud()
        pcl_data.points = o3d.utility.Vector3dVector(points)
        return pcl_data

    def process_and_publish(self):
        if self.point_cloud is None or self.ellipsoid_marker is None:
            return

        # Extract ellipsoid parameters from the marker
        center = np.array([self.ellipsoid_marker.pose.position.x,
                           self.ellipsoid_marker.pose.position.y,
                           self.ellipsoid_marker.pose.position.z])

        scale = np.array([self.ellipsoid_marker.scale.x / 2,
                          self.ellipsoid_marker.scale.y / 2,
                          self.ellipsoid_marker.scale.z / 2])

        # Create a copy of the point cloud and move it towards the ellipsoid center
        moved_point_cloud = self.point_cloud.translate(np.asarray([0,0,1]) * 0.05, relative=True)

        # Combine the original and moved point clouds
        combined_point_cloud = self.point_cloud + moved_point_cloud

        # Compute the concave hull
        concave_hull, _ = combined_point_cloud.compute_convex_hull()

        # Downsample the concave hull to create the filled point cloud
        num_points = 100  # Set the number of points you want in the downsampled cloud
        filled_point_cloud = concave_hull.sample_points_poisson_disk(num_points)

        # Convert to PointCloud2 and publish
        self.publish_filled_point_cloud(filled_point_cloud)

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

        cloud_data = np.asarray(filled_points.points).astype(np.float32)
        cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        self.filled_cloud_pub.publish(cloud_msg)

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
