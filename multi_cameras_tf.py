#!/usr/bin/env python3
import rospy
import tf
import geometry_msgs.msg

def publish_transforms(broadcaster):
    # camera_link1의 변환을 생성하고 브로드캐스트합니다
    camera_link1_trans = geometry_msgs.msg.TransformStamped()
    camera_link1_trans.header.stamp = rospy.Time.now()
    camera_link1_trans.header.frame_id = "world"
    camera_link1_trans.child_frame_id = "camera1_link"
    camera_link1_trans.transform.translation.x = 0.0
    camera_link1_trans.transform.translation.y = 0.0
    camera_link1_trans.transform.translation.z = 0.0
    camera_link1_trans.transform.rotation.x = 0.0
    camera_link1_trans.transform.rotation.y = 0.0
    camera_link1_trans.transform.rotation.z = 0.0
    camera_link1_trans.transform.rotation.w = 1.0

    # camera_link2의 변환을 생성하고 브로드캐스트합니다
    camera_link2_trans = geometry_msgs.msg.TransformStamped()
    camera_link2_trans.header.stamp = rospy.Time.now()
    camera_link2_trans.header.frame_id = "world"
    camera_link2_trans.child_frame_id = "camera2_link"
    camera_link2_trans.transform.translation.x = 0.05
    camera_link2_trans.transform.translation.y = 0.00
    camera_link2_trans.transform.translation.z = -0.05
    camera_link2_trans.transform.rotation.x = 0.0
    camera_link2_trans.transform.rotation.y = 0.0
    camera_link2_trans.transform.rotation.z = 0.0
    camera_link2_trans.transform.rotation.w = 1.0

    # camera_link1과 camera_link2 사이의 변환을 계산하고 브로드캐스트합니다
    camera_link1_to_link2_trans = geometry_msgs.msg.TransformStamped()
    camera_link1_to_link2_trans.header.stamp = rospy.Time.now()
    camera_link1_to_link2_trans.header.frame_id = "camera1_link"
    camera_link1_to_link2_trans.child_frame_id = "camera2_link"
    camera_link1_to_link2_trans.transform.translation.x = 0.0
    camera_link1_to_link2_trans.transform.translation.y = 0.0
    camera_link1_to_link2_trans.transform.translation.z = 0.05
    camera_link1_to_link2_trans.transform.rotation.x = 0.0
    camera_link1_to_link2_trans.transform.rotation.y = 0.0
    camera_link1_to_link2_trans.transform.rotation.z = 0.0
    camera_link1_to_link2_trans.transform.rotation.w = 1.0

    # 브로드캐스트
    broadcaster.sendTransformMessage(camera_link1_trans)
    broadcaster.sendTransformMessage(camera_link2_trans)
    # broadcaster.sendTransformMessage(camera_link1_to_link2_trans)

if __name__ == '__main__':
    rospy.init_node('camera_tf_broadcaster')
    broadcaster = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        publish_transforms(broadcaster)
        rate.sleep()
