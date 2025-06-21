import rospy
from visualization_msgs.msg import Marker

def publish_transparent_marker():
    rospy.init_node('transparent_marker_publisher', anonymous=True)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "transparent_cube"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    # Set the scale of the marker
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    # Set the pose of the marker
    marker.pose.position.x = 0.5
    marker.pose.position.y = 0.5
    marker.pose.position.z = 1.5
    marker.pose.orientation.w = 1.0

    # Set the color (including transparency)
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 0.1  # Transparency

    # marker.lifetime = rospy.Duration()

    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rospy.sleep(1)

if __name__ == '__main__':
    try:
        publish_transparent_marker()
    except rospy.ROSInterruptException:
        pass
