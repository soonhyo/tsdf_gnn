import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import tf.transformations as tf_trans

class PointCloudProcessor:
    def __init__(self):
        rospy.init_node('ellipsoid_fitter', anonymous=True)
        self.point_cloud_sub = rospy.Subscriber('/output/point_cloud', PointCloud2, self.point_cloud_callback)
        self.pose_sub = rospy.Subscriber('/output/pose', PoseStamped, self.pose_callback)
        self.marker_pub = rospy.Publisher('/ellipsoid_marker', Marker, queue_size=10)
        self.point_cloud = None
        self.current_pose = None

    def point_cloud_callback(self, msg):
        # Convert PointCloud2 message to Open3D PointCloud
        self.point_cloud = self.pointcloud2_to_open3d(msg)
        self.publish_ellipsoid()

    def pose_callback(self, msg):
        # Save the current pose
        self.current_pose = msg.pose

    def pointcloud2_to_open3d(self, cloud_msg):
        # Convert sensor_msgs/PointCloud2 to Open3D PointCloud
        points = np.array(list(pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))))
        pcl_data = o3d.geometry.PointCloud()
        pcl_data.points = o3d.utility.Vector3dVector(points)
        return pcl_data

    def adjust_axes_with_camera_direction(self, axes):
        # Assume camera is looking in the  z-direction of the camera frame
        camera_direction = np.array([0, 0, 1])

        # Adjust each axis to align with the camera direction
        for i in range(3):
            if np.dot(axes[i], camera_direction) < 0:
                axes[i] = -axes[i]
        return axes

    def publish_ellipsoid(self):
        if self.point_cloud is None or self.current_pose is None:
            return

        # Extract points from Open3D PointCloud
        points = np.asarray(self.point_cloud.points)

        # Fit an ellipsoid using PCA
        pca = PCA(n_components=3)
        pca.fit(points)
        center = pca.mean_
        axes = pca.components_
        lengths = [0.1, 0.12, 0.1]

        # Adjust axes to maintain consistency with the camera direction
        axes = self.adjust_axes_with_camera_direction(axes)

        translation_axes = 0.1 * axes[2]
        center += translation_axes

        # Create Marker message for the ellipsoid
        marker = Marker()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ellipsoid"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]

        # Reflect the orientation over the xy plane
        original_orientation = [
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ]

        # Define the quaternion for 180 degree rotation about the x axis
        reflect_quaternion = tf_trans.quaternion_from_euler(0, 0, np.pi)

        # Reflect the original orientation
        reflected_orientation = tf_trans.quaternion_multiply(original_orientation, reflect_quaternion)

        marker.pose.orientation.x = reflected_orientation[0]
        marker.pose.orientation.y = reflected_orientation[1]
        marker.pose.orientation.z = reflected_orientation[2]
        marker.pose.orientation.w = reflected_orientation[3]

        marker.scale.x = lengths[0] * 2  # Diameter along x-axis
        marker.scale.y = lengths[1] * 2  # Diameter along y-axis
        marker.scale.z = lengths[2] * 2  # Diameter along z-axis

        marker.color.a = 0.5  # Transparency
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Publish the Marker message
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    try:
        processor = PointCloudProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
