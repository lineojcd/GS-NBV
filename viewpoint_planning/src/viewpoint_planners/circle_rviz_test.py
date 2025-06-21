#!/usr/bin/env python

import rospy
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def draw_circle(radius, center, resolution=100):
    # Create a marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "circle"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.005  # Line width
    # marker.color.a = 1.0  # Don't forget to set the alpha!
    marker.color.a = 0.7  # Don't forget to set the alpha!
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    # Generate points on the circle
    for i in range(resolution + 1):
        angle = 2 * math.pi * i / resolution
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        marker.points.append(point)
    
    return marker

def circle_marker_publisher():
    rospy.init_node('circle_marker_publisher')
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker = draw_circle(radius=1.0, center=(0.0, 0.0, 0.0))
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        circle_marker_publisher()
    except rospy.ROSInterruptException:
        pass
