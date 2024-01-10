#!/usr/bin/env python
import rospy
import tf
import tf2_ros
import geometry_msgs.msg
import numpy as np
def publish_camera_tf():
    rospy.init_node('camera_tf_broadcaster')

    br = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        # 카메라와 베이스 사이의 변환을 정의합니다.
        # 여기서는 카메라가 베이스에 대해 1미터 앞쪽에 위치하고, 45도 기울어져 있다고 가정합니다.
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "camera1_link"  # 베이스 프레임
        t.child_frame_id = "camera2_link"  # 카메라 프레임

        # # 위치 설정
        # t.transform.translation.x = 1.3
        # t.transform.translation.y = -0.05
        # t.transform.translation.z = 0.1

        # # 방향 설정 (여기서는 45도 회전)
        # q = tf.transformations.quaternion_from_euler(np.pi, -np.pi, 0)  # 45도는 라디안으로 약 0.785398

        # 위치 설정
        t.transform.translation.x = 0
        t.transform.translation.y = 0
        t.transform.translation.z = 0.3

        # 방향 설정 (여기서는 45도 회전)
        # q = tf.transformations.quaternion_from_euler(np.pi, -np.pi, 0)  # 45도는 라디안으로 약 0.785398
        q = [0, 0, 0, 1]
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # 변환 발행
        br.sendTransform(t)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_camera_tf()
    except rospy.ROSInterruptException:
        pass
