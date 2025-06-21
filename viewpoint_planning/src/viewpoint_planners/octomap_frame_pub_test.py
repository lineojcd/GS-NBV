#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def publish_voxel_frame():
    rospy.init_node('voxel_frame_visualizer', anonymous=True)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    # Voxel frame origin
    origin = (0.2600, -0.5400, 0.1500)
    
    # Length of each voxel
    voxel_length = 0.1
    
    # Publish each axis as a separate marker
    def create_axis_marker(axis, color, start, end):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "voxel_frame"
        marker.id = axis
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set start and end points
        marker.points = [Point(*start), Point(*end)]
        
        # Set the scale of the arrow
        marker.scale.x = 0.01  # Shaft diameter
        marker.scale.y = 0.02  # Head diameter
        marker.scale.z = 0.0   # Head length (not used)
        
        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        
        return marker

    # X-axis in red
    x_axis_marker = create_axis_marker(
        axis=0,
        color=(1.0, 0.0, 0.0),
        start=origin,
        end=(origin[0] + voxel_length, origin[1], origin[2])
    )

    # Y-axis in green
    y_axis_marker = create_axis_marker(
        axis=1,
        color=(0.0, 1.0, 0.0),
        start=origin,
        end=(origin[0], origin[1] + voxel_length, origin[2])
    )

    # Z-axis in blue
    z_axis_marker = create_axis_marker(
        axis=2,
        color=(0.0, 0.0, 1.0),
        start=origin,
        end=(origin[0], origin[1], origin[2] + voxel_length)
    )

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(x_axis_marker)
        marker_pub.publish(y_axis_marker)
        marker_pub.publish(z_axis_marker)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_voxel_frame()
    except rospy.ROSInterruptException:
        pass
