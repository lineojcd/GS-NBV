#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler
import numpy as np

def move_to_predefined_pose():
    # Initialize the moveit_commander and rospy
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_j2n6s300_to_pose', anonymous=True)

    # Instantiate a RobotCommander object. This object is an interface to the robot as a whole.
    robot = moveit_commander.RobotCommander()

    # Instantiate a PlanningSceneInterface object. This object is an interface to the world surrounding the robot.
    scene = moveit_commander.PlanningSceneInterface()

    # Instantiate a MoveGroupCommander object. This object is an interface to one group of joints.
    group_name = "arm"  # Change this if your MoveGroup name is different
    move_group = moveit_commander.MoveGroupCommander(group_name)
    
    # Increase the allowed planning time
    max_planning_time = 15.0
    move_group.set_planning_time(max_planning_time)  # Increase this value to give more time for planning

    # We can get the name of the reference frame for this robot
    planning_frame = move_group.get_planning_frame()
    rospy.loginfo("Reference frame: %s" % planning_frame)  # world

    # We can also print the name of the end-effector link for this group
    eef_link = move_group.get_end_effector_link()
    rospy.loginfo("End effector link: %s" % eef_link)      # j2n6s300_end_effector

    # Define a target pose
    pose_goal = geometry_msgs.msg.PoseStamped()
    pose_goal.header.frame_id = planning_frame
    pose_goal.header.stamp = rospy.Time.now()

    while True:
        # Prompt user for input
        user_input = input("Enter 4 float numbers for quaternion (x, y, z, w) separated by spaces, or type 'quit' to exit: ")
        
        if user_input.lower() == "quit":
            rospy.loginfo("Quitting the program...")
            moveit_commander.roscpp_shutdown()
            sys.exit(0)

        try:
            # Convert input to a numpy array
            q = np.array([float(num) for num in user_input.split()])
            
            if len(q) != 4:
                raise ValueError("Please enter exactly 4 float numbers.")
                
            # Print the quaternion
            print("q is :", q)
            
            # Assign the input quaternion to the pose
            pose_goal.pose.position.x = 0.2
            pose_goal.pose.position.y = 0.2
            pose_goal.pose.position.z = 0.6
            pose_goal.pose.orientation.x = q[0]
            pose_goal.pose.orientation.y = q[1]
            pose_goal.pose.orientation.z = q[2]
            pose_goal.pose.orientation.w = q[3]
            print("pose_goal is :", pose_goal)
            
            # Set the goal pose for the end effector
            move_group.set_pose_target(pose_goal)
            
            # Plan to the new pose
            plan = move_group.plan()
            plan_success = plan[0]
            joint_trajectory = plan[1]
            print("plan_success is:",plan_success)
            
            if plan_success:
                # Execute the plan
                go = move_group.go(wait=True)
                rospy.loginfo("Move successful: %s" % go)
                
                # Ensure there is no residual movement
                move_group.stop()
                # Clear the pose target after planning with it
                move_group.clear_pose_targets()

                # Print the final pose of the end effector
                # current_state = move_group.get_current_pose()
                current_state = robot.get_current_state()
                rospy.loginfo("Current state: %s" % current_state)
            else:
                rospy.logwarn("Planning failed. No execution performed.")
                print("no result")
        
        except ValueError as e:
            rospy.logwarn(str(e))
            continue

    # Shut down moveit_commander
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        move_to_predefined_pose()
    except rospy.ROSInterruptException:
        pass
