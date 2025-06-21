#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker

def publish_point():
    rospy.init_node('pink_point_visualizer', anonymous=True)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    # Create a marker for the point
    marker = Marker()
    marker.header.frame_id = "world"  # You can change this to the appropriate frame
    marker.header.stamp = rospy.Time.now()

    marker.ns = "pink_point"
    marker.id = 0
    marker.type = Marker.SPHERE  # Use a sphere to represent a point
    marker.action = Marker.ADD

    # Set the position of the point
    marker.pose.position.x = 0.50065
    marker.pose.position.y = -0.24662
    marker.pose.position.y = -0.26
    marker.pose.position.z = 0.47821
 
    # Set orientation to identity (no rotation)
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Set the scale of the sphere (this determines the size of the point)
    marker.scale.x = 0.01 # Adjust as needed
    marker.scale.y = 0.01
    marker.scale.z = 0.01

    # Set the color to pink (RGBA)
    marker.color.r = 1.0
    marker.color.g = 0.4
    marker.color.b = 0.7
    marker.color.a = 1.0  # Ensure the alpha is 1 for full opacity

    # Set the marker lifetime (how long it will be visible)
    marker.lifetime = rospy.Duration()

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_point()
    except rospy.ROSInterruptException:
        pass
