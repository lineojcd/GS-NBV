#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def draw_sphere():
    rospy.init_node('draw_sphere_node')

    # Create a publisher for the Marker
    marker_pub = rospy.Publisher('/arm_workspace', Marker, queue_size=10)

    # Define the marker
    marker = Marker()
    marker.header.frame_id = "world"  # Change this to your fixed frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "arm_workspace"
    marker.id = 1
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    # Define the position (center of the sphere)
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.2755 + 0.66 / 2  # Move the center up by half the radius

    # Define the orientation
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Define the scale (only draw the upper hemisphere)
    marker.scale.x = 1.32  # Increased scale
    marker.scale.y = 1.32
    marker.scale.z = 0.66

    # Define the color
    marker.color.r = 0.5
    marker.color.g = 0.0
    marker.color.b = 0.5
    marker.color.a = 0.3  # 60% transparency

    marker.lifetime = rospy.Duration(0)  # 0 means infinite duration
    # Publish the marker
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rate.sleep()

    # marker_pub.publish(marker)

    rospy.loginfo("Sphere marker published.")

if __name__ == '__main__':
    try:
        draw_sphere()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
