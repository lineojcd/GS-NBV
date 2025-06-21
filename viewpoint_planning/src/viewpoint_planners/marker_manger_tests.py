import rospy
from visualization_msgs.msg import Marker, MarkerArray

class MarkerManager:
    def __init__(self, publisher):
        self.publisher = publisher
        self.markers = {}
        self.next_id = 0

    def add_marker(self, marker):
        """
        Adds or updates a marker in the system.
        """
        if marker.id in self.markers:
            rospy.loginfo("Updating existing marker: %s", marker.id)
        else:
            marker.id = self.next_id
            self.next_id += 1
            rospy.loginfo("Adding new marker: %s", marker.id)

        # Set common fields
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD

        # Store marker
        self.markers[marker.id] = marker

        # Publish marker
        self.publish_markers()

    def delete_marker(self, marker_id):
        """
        Deletes a marker by its ID.
        """
        if marker_id in self.markers:
            marker = self.markers.pop(marker_id)
            marker.action = Marker.DELETE
            self.publisher.publish(marker)
            rospy.loginfo("Deleted marker: %s", marker_id)
        else:
            rospy.logwarn("Attempted to delete non-existent marker ID: %s", marker_id)

    def publish_markers(self):
        """
        Publish all markers.
        """
        array = MarkerArray()
        array.markers = list(self.markers.values())
        self.publisher.publish(array)

# Example usage
if __name__ == "__main__":
    rospy.init_node("marker_manager_example")
    marker_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
    manager = MarkerManager(marker_pub)
    rospy.sleep(1)  # Allow time for ROS to set up the publisher

    # Example adding a marker
    new_marker = Marker(type=Marker.SPHERE, scale=Vector3(0.1, 0.1, 0.1), color=ColorRGBA(1, 0, 0, 1))
    manager.add_marker(new_marker)

    # Example deleting a marker
    # manager.delete_marker(0)
