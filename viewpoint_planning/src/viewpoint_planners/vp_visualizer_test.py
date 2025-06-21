import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import ColorRGBA
import numpy as np
class Visualizer:
    def __init__(self):
        rospy.init_node('viewpoint_visualizer', anonymous=True)
        self.viewpoint_pub = rospy.Publisher("viewpoint", Marker, queue_size=1)
        self.world_frame_id = "world"  # Make sure this matches your RViz fixed frame

    def visualize_viewpoint(self, viewpoint: Pose) -> None:
        """
        Visualize the viewpoint in Rviz as a Marker message
        :param viewpoint: Pose of the viewpoint
        """
        print("viewpoint is:", viewpoint)
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "viewpoint"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = viewpoint
        marker.scale = Vector3(1, 0.1, 0.1)  # Scale the arrow appropriately
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # Red color
        self.viewpoint_pub.publish(marker)

if __name__ == "__main__":
    # visualizer = Visualizer()
    
    # # Create a sample viewpoint Pose
    # sample_pose = Pose()
    # sample_pose.position.x = 0.0
    # sample_pose.position.y = 0.0
    # sample_pose.position.z = 1.0
    # sample_pose.orientation.x = 0.0
    # sample_pose.orientation.y = 0.0
    # sample_pose.orientation.z = 0.0
    # sample_pose.orientation.w = 1.0

    # rate = rospy.Rate(10)  # 10 Hz
    # while not rospy.is_shutdown():
    #     visualizer.visualize_viewpoint(sample_pose)
    #     rate.sleep()

    fruit_position= [0.50066,    -0.24676,     0.47844]
    viewpoints=  [0.23276,    -0.35575,     0.55815,     0.78072,     0.15274,     0.59466,    -0.11634]
    vplst = [np.array(viewpoints)]
    vplst = np.array(viewpoints)
    for idx, vp in enumerate(vplst):
        dir_vec =  fruit_position - vp[:3]
    print(dir_vec)