import rospy
from geometry_msgs.msg import PoseStamped
import argparse
import time

class Jaco2ControlClient:
    """
    Sends pose commands to move the Jaco2 arm to a specified position and orientation.
    """

    def __init__(self):
        self.prefix = 'j2n6s300'
        self.pose_topic = '/' + self.prefix + '/in/cartesian_pose'
        self.pose_topic  = '/j2n6s300/effort_joint_trajectory_controller/command'


    def move_to_pose(self, position, orientation):
        """
        Move the Jaco2 arm to the given Cartesian pose.
        
        Parameters:
            position (tuple): The target position as a tuple (x, y, z) in meters.
            orientation (tuple): The target orientation as a quaternion (x, y, z, w).
        """
        pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"  # Ensure this matches the expected frame_id in your setup

        # Set position
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]

        # Set orientation
        pose_msg.pose.orientation.x = orientation[0]
        pose_msg.pose.orientation.y = orientation[1]
        pose_msg.pose.orientation.z = orientation[2]
        pose_msg.pose.orientation.w = orientation[3]

        rate = rospy.Rate(10)  # 10 Hz
        for _ in range(10):  # Publish the pose message multiple times
            pub.publish(pose_msg)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('jaco2_pose_commander')

    # Example position and orientation
    position = (0.2, -0.2, 0.5)  # x, y, z in meters
    orientation = (0, 0, 0, 1)  # Quaternion x, y, z, w

    controller = Jaco2ControlClient()
    controller.move_to_pose(position, orientation)
