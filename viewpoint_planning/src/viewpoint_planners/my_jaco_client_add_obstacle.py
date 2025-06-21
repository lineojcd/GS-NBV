#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from visualization_msgs.msg import Marker
from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler
import numpy as np

def remove_all_obstacles(scene):
    # Remove all objects from the planning scene
    scene.remove_world_object()
    rospy.sleep(1)  # Give time for the objects to be removed

def remove_obstacle(scene, box_name):
    # Remove the box with the specified name from the planning scene
    scene.remove_world_object(box_name)
    rospy.sleep(1)  # Give time for the object to be removed

def add_obstacle(scene, planning_frame):
    # Define a box obstacle
    box_pose1 = geometry_msgs.msg.PoseStamped()
    box_pose1.header.frame_id = planning_frame
    box_pose1.pose.position.x = 0.3  # Adjust the position as needed
    box_pose1.pose.position.y = 0.0
    box_pose1.pose.position.z = 0.5
    box_pose1.pose.orientation.w = 1.0
    box_name1 = "obstacle_box"
    box_size1 = (0.2, 0.2, 0.2)  # Adjust the size as needed
    # Add the box to the planning scene
    scene.add_box(box_name1, box_pose1, size=box_size1)
    
    box_pose2 = geometry_msgs.msg.PoseStamped()
    box_pose2.header.frame_id = planning_frame
    box_pose2.pose.position.x = 0.5  # Adjust the position as needed
    box_pose2.pose.position.y = 0.0
    box_pose2.pose.position.z = 0.6
    box_pose2.pose.orientation.w = 1.0
    box_name2 = "obstacle_box2"
    box_size2 = (0.2, 0.2, 0.2)  # Adjust the size as needed
    scene.add_box(box_name2, box_pose2, size=box_size2)
    rospy.sleep(2)  # Allow some time for the box to be added to the scene
    
    # The below part is useless
    if False:
        # Create a marker for visualization in Rviz
        marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        
        marker = Marker()
        marker.header.frame_id = planning_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacle"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = box_pose.pose
        marker.scale.x = box_size[0]
        marker.scale.y = box_size[1]
        marker.scale.z = box_size[2]
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.1  # Set transparency here (0.0 fully transparent, 1.0 fully opaque)
        # marker.lifetime = rospy.Duration()
        
        marker_pub.publish(marker)

        rospy.sleep(2)  # Allow some time for the marker to be displayed


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
    
    # Add an obstacle to the scene
    add_obstacle(scene, planning_frame)

    # Define a target pose
    pose_goal = geometry_msgs.msg.PoseStamped()
    pose_goal.header.frame_id = planning_frame
    pose_goal.header.stamp = rospy.Time.now()

    while True:
        # Prompt user for input
        user_input = input("Enter 7 float numbers for quaternion (x, y, z, w) separated by spaces, or type 'quit' to exit: ")
        
        if user_input.lower() == "quit":
            rospy.loginfo("Quitting the program...")
            
            # Later, when you want to remove the boxes:
            # remove_obstacle(scene, "obstacle_box_1")
            # remove_obstacle(scene, "obstacle_box_2")
            remove_all_obstacles(scene)
            moveit_commander.roscpp_shutdown()
            sys.exit(0)

        try:
            # Convert input to a numpy array
            q = np.array([float(num) for num in user_input.split()])
            
            if len(q) != 7 :
                raise ValueError("Please enter exactly 7 float numbers.")
                
            # Print the quaternion
            print("q is :", q)
            
            # Assign the input quaternion to the pose
            # pose_goal.pose.position.x = 0.2
            # pose_goal.pose.position.y = 0.2
            # pose_goal.pose.position.z = 0.6
            pose_goal.pose.position.x = q[0]
            pose_goal.pose.position.y = q[1]
            pose_goal.pose.position.z = q[2]
            pose_goal.pose.orientation.x = q[3]
            pose_goal.pose.orientation.y = q[4]
            pose_goal.pose.orientation.z = q[5]
            pose_goal.pose.orientation.w = q[6]
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
