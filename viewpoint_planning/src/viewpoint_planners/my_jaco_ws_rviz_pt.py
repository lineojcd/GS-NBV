#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math

def draw_top_half_sphere():
    rospy.init_node('draw_top_half_sphere_node')
    marker_pub = rospy.Publisher('/arm_workspace', Marker, queue_size=10)
    # Define the marker for points
    marker = Marker()
    marker.header.frame_id = "world"  # Ensure this matches your RViz fixed frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "arm_workspace"
    marker.id = 1
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.005  # Size of the points
    marker.scale.y = 0.005
    # marker.scale.y = 0.02
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.3
    radius = 0.66
    center_z = 0.2755
    # Generate points for the top half of the sphere
    for theta in range(0, 181, 3):  # Polar angle from 0 to 180 degrees
        for phi in range(0, 361, 3):  # Azimuthal angle from 0 to 360 degrees
            theta_rad = math.radians(theta)
            phi_rad = math.radians(phi)
            
            x = radius * math.sin(theta_rad) * math.cos(phi_rad)
            y = radius * math.sin(theta_rad) * math.sin(phi_rad)
            z = center_z + radius * math.cos(theta_rad)

            if z >= center_z:  # Only points above the center
                point = Point()
                point.x = x
                point.y = y
                point.z = z
                marker.points.append(point)

    marker.lifetime = rospy.Duration(0)  # Infinite duration

    # Publish the marker
    # for i in range(10):
    marker_pub.publish(marker)

    rospy.loginfo("Workspace published.")
    
    rate = rospy.Rate(0.1)  # Publish at 10 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == "__main__":
    try:
        draw_top_half_sphere()
    except rospy.ROSInterruptException:
        pass
