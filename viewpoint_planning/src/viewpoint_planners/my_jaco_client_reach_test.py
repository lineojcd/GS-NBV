#!/usr/bin/env python

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import math
import sys


# Void code

def get_max_reach():
    # Initialize the moveit_commander and rospy node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('test_max_reach', anonymous=True)
    
    # Initialize the MoveGroupCommander for the arm
    group_name = "arm"  # Replace with the correct group name for your setup
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Get the current joint values and set extreme positions
    joint_values = move_group.get_current_joint_values()

    print("joint_values are",joint_values)
    
    # Assuming you want to extend each joint to its max
    for i in range(len(joint_values)):
        joint_values[i] = move_group.get_joint_limits()[i].max_position

    print("joint_values are",joint_values)
    
    # Set the new joint values
    move_group.set_joint_value_target(joint_values)

    # Plan and move the arm
    move_group.go(wait=True)

    # Get the end-effector pose at maximum reach
    max_reach_pose = move_group.get_current_pose().pose
    rospy.loginfo("Max reach pose: {}".format(max_reach_pose))

    # Clean up
    move_group.stop()
    move_group.clear_pose_targets()
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    try:
        get_max_reach()
    except rospy.ROSInterruptException:
        pass
