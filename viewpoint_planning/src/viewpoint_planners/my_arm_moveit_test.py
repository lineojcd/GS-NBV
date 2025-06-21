#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Bool
import math
class PoseController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('pose_controller')

        # Publisher for sending poses
        self.pose_pub = rospy.Publisher('/vp_pose', Pose, queue_size=1)

        # Subscriber to receive move status
        self.status_sub = rospy.Subscriber('/arm_move_status', Bool, self.status_callback)

        # Define some realistic dummy poses
        self.dummy_poses = [
            Pose(position=Point(x=0.5, y=0.3, z=0.1), orientation=Quaternion(*quaternion_from_euler(0, 0, math.pi/2))),
            Pose(position=Point(x=0.4, y=0.1, z=0.2), orientation=Quaternion(*quaternion_from_euler(0, 0, math.pi/4))),
            Pose(position=Point(x=0.3, y=0.2, z=0.2), orientation=Quaternion(*quaternion_from_euler(0, 0, math.pi/6))),
            Pose(position=Point(x=0.2, y=0.3, z=0.2), orientation=Quaternion(*quaternion_from_euler(0, 0, math.pi/8))),
            Pose(position=Point(x=0.1, y=0.4, z=0.2), orientation=Quaternion(*quaternion_from_euler(0, 0, 0)))
        ]
        # Track the index of the current pose to publish
        self.current_pose_index = 0

        # Flag to control the publishing loop
        self.continue_publishing = True

    def publish_next_pose(self):
        """Publish the next pose from the list."""
        # if self.current_pose_index < len(self.dummy_poses) and self.continue_publishing :
        if self.current_pose_index < len(self.dummy_poses) :
            pose = self.dummy_poses[0]
            self.pose_pub.publish(pose)
            # self.continue_publishing = False
            rospy.loginfo("Published pose {}".format(self.current_pose_index + 1))
        else:
            rospy.loginfo("All poses have been published.")
            self.continue_publishing = False

    def status_callback(self, msg):
        """Callback for processing the move status."""
        user_input = input(">>>>>>>>>>>Continue (y/n)? ")
        if user_input.lower() == 'y':
            print("msg.data is:", msg.data)
            if msg.data:
                rospy.loginfo("Move was successful, advancing to the next pose.")
                self.current_pose_index += 1
                # self.continue_publishing = True
                self.publish_next_pose()
            else:
                rospy.loginfo("Move was not successful. Asking user to continue or quit.")
                user_input = input("Continue (y/n)? ")
                if user_input.lower() != 'y':
                    self.continue_publishing = False

    def run(self):
        """Run the pose publishing loop."""
        while not rospy.is_shutdown():
            # if self.current_pose_index == 0:  # To publish the first pose
            if self.continue_publishing:  # To publish the first pose
                self.publish_next_pose()
            rospy.sleep(1)  # Sleep to wait for callback to be processed

if __name__ == '__main__':
    controller = PoseController()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
