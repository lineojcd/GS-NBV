#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
import time

class Jaco2ControlClient:
    def __init__(self,obstacle_center=[0.9, 0.15 , 0.0], standalone=True):
        # Initialize the moveit_commander and rospy
        moveit_commander.roscpp_initialize(sys.argv)
        if standalone:
            rospy.init_node('move_j2n6s300_to_pose', anonymous=True)

        self.arm_max_reach = rospy.get_param("arm_max_reach", "0.66")
        self.setting_workspace_time = rospy.get_param("setting_workspace_time", "2")
        self.tree_obstacle_range = [float(x) for x in rospy.get_param('tree_obstacle_range').split()]
        
        self.obstacle_center=obstacle_center
        # Instantiate a RobotCommander object. This object is an interface to the robot as a whole.
        self.robot = moveit_commander.RobotCommander()

        # Instantiate a PlanningSceneInterface object. This object is an interface to the world surrounding the robot.
        self.scene = moveit_commander.PlanningSceneInterface()

        # Instantiate a MoveGroupCommander object. This object is an interface to one group of joints.
        self.group_name = rospy.get_param('group_name', 'arm') 
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # Increase the allowed planning time
        self.max_planning_time = rospy.get_param("max_planning_time", "30")
        self.move_group.set_planning_time(self.max_planning_time)  # Increase this value to give more time for planning

        # We can get the name of the reference frame for this robot
        self.planning_frame = self.move_group.get_planning_frame()
        rospy.loginfo("Reference frame: %s" % self.planning_frame)  # world
        
        # We can also print the name of the end-effector link for this group
        self.eef_link = self.move_group.get_end_effector_link()
        rospy.loginfo("End effector link: %s" % self.eef_link)      # j2n6s300_end_effector
        
        self.add_tree_obstacle()
        # self.add_obstacle_test()
        self.show_workspace()
    
    def remove_all_obstacles(self):
        # Remove all objects from the planning scene
        self.scene.remove_world_object()
        rospy.sleep(1)  # Give time for the objects to be removed
    
    def remove_obstacle(self, box_name):
        # Remove the box with the specified name from the planning scene
        self.scene.remove_world_object(box_name)
        rospy.sleep(1)  # Give time for the object to be removed
    
    def add_tree_obstacle(self):
        rospy.loginfo("Add tree obstacle")
        # Define a box obstacle to represent the tree
        box_tree_pose = geometry_msgs.msg.PoseStamped()
        box_tree_pose.header.frame_id = self.planning_frame
        box_tree_pose.pose.position.x = self.obstacle_center[0]  # Adjust the position as needed
        box_tree_pose.pose.position.y = self.obstacle_center[1] 
        box_tree_pose.pose.position.z = self.obstacle_center[2] + 0.42
        box_tree_pose.pose.orientation.w = 1.0
        box_tree_name = "tree_obstacle"
        # box_tree_size = (0.22, 0.32, 0.42)  
        box_tree_size = self.tree_obstacle_range  
        # print("box_tree_size is :",box_tree_size )
        # Add the box to the planning scene
        self.scene.add_box(box_tree_name, box_tree_pose, size=box_tree_size)
        rospy.sleep(1)  # Allow some time for the box to be added to the scene
    
    def add_obstacle_test(self):
        # Define a box obstacle
        box_pose1 = geometry_msgs.msg.PoseStamped()
        box_pose1.header.frame_id = self.planning_frame
        box_pose1.pose.position.x = 0.3  # Adjust the position as needed
        box_pose1.pose.position.y = 0.0
        box_pose1.pose.position.z = 0.5
        box_pose1.pose.orientation.w = 1.0
        box_name1 = "obstacle_box"
        box_size1 = (0.2, 0.2, 0.2)  # Adjust the size as needed
        box_pose2 = geometry_msgs.msg.PoseStamped()
        box_pose2.header.frame_id = self.planning_frame
        box_pose2.pose.position.x = 0.5  # Adjust the position as needed
        box_pose2.pose.position.y = 0.0
        box_pose2.pose.position.z = 0.6
        box_pose2.pose.orientation.w = 1.0
        box_name2 = "obstacle_box2"
        box_size2 = (0.2, 0.2, 0.2)  # Adjust the size as needed
        # Add the box to the planning scene
        self.scene.add_box(box_name1, box_pose1, size=box_size1)
        self.scene.add_box(box_name2, box_pose2, size=box_size2)
        rospy.sleep(2)  # Allow some time for the box to be added to the scene
    
    def check_plan_exists(self, pose):
        try:
            # Define a target pose
            pose_goal = geometry_msgs.msg.PoseStamped()
            pose_goal.header.frame_id = self.planning_frame
            pose_goal.header.stamp = rospy.Time.now()
            
            # Assign the input quaternion to the pose
            pose_goal.pose.position.x = pose[0]
            pose_goal.pose.position.y = pose[1]
            pose_goal.pose.position.z = pose[2]
            pose_goal.pose.orientation.x = pose[3]
            pose_goal.pose.orientation.y = pose[4]
            pose_goal.pose.orientation.z = pose[5]
            pose_goal.pose.orientation.w = pose[6]
            
            # Set the goal pose for the end effector
            self.move_group.set_pose_target(pose_goal)
            
            # Plan to the new pose
            plan = self.move_group.plan()
            plan_success = plan[0]
            joint_trajectory = plan[1]
            rospy.loginfo(f"Plan exists: {plan_success}")
            return plan_success
        except rospy.ServiceException as e:
            print("[Jaco2ControlClient] Service call failed: %s" % e)
            return False
    
    def move_arm_to_pose(self, pose, verbose=False):
        try:
            # Define a target pose
            pose_goal = geometry_msgs.msg.PoseStamped()
            pose_goal.header.frame_id = self.planning_frame
            pose_goal.header.stamp = rospy.Time.now()
            
            # Assign the input quaternion to the pose
            pose_goal.pose.position.x = pose[0]
            pose_goal.pose.position.y = pose[1]
            pose_goal.pose.position.z = pose[2]
            pose_goal.pose.orientation.x = pose[3]
            pose_goal.pose.orientation.y = pose[4]
            pose_goal.pose.orientation.z = pose[5]
            pose_goal.pose.orientation.w = pose[6]
            
            # Set the goal pose for the end effector
            self.move_group.set_pose_target(pose_goal)
            
            # Plan to the new pose
            plan = self.move_group.plan()
            plan_success = plan[0]
            joint_trajectory = plan[1]
            rospy.loginfo(f"Plan exists: {plan_success}")
            
            if plan_success:
                # Execute the plan
                result = self.move_group.go(wait=True)
                rospy.loginfo("Move successful: %s" % result)
                
                # Ensure there is no residual movement
                self.move_group.stop()
                # Clear the pose target after planning with it
                self.move_group.clear_pose_targets()

                # Print the final pose of the end effector
                # current_state = move_group.get_current_pose()
                if verbose:
                    current_state = self.robot.get_current_state()
                    rospy.loginfo("Current state: %s" % current_state)
                
                return plan_success
            else:
                rospy.logwarn("Planning failed. No execution performed.")
                return False
            
        except rospy.ServiceException as e:
            print("[Jaco2ControlClient] Service call failed: %s" % e)
            return False
        
    def check_trajectory_exist(self, pose, verbose=False):
        try:
            pose_goal = geometry_msgs.msg.PoseStamped()
            pose_goal.header.frame_id = self.planning_frame
            pose_goal.header.stamp = rospy.Time.now()
            pose_goal.pose.position.x = pose[0]
            pose_goal.pose.position.y = pose[1]
            pose_goal.pose.position.z = pose[2]
            pose_goal.pose.orientation.x = pose[3]
            pose_goal.pose.orientation.y = pose[4]
            pose_goal.pose.orientation.z = pose[5]
            pose_goal.pose.orientation.w = pose[6]
            # Set the goal pose for the end effector
            self.move_group.set_pose_target(pose_goal)
            plan = self.move_group.plan()
            plan_success = plan[0]
            rospy.loginfo(f"Plan exists: {plan_success}")
            return plan_success
        except rospy.ServiceException as e:
            print("[Jaco2ControlClient] Service call failed: %s" % e)
            return False
    
    def show_workspace(self):
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
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3
        radius = self.arm_max_reach
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
        # marker_pub.publish(marker)
        start_time = time.time()
        rate = rospy.Rate(0.1)  # Publish at 10 Hz
        while not rospy.is_shutdown():
            marker_pub.publish(marker)
            if time.time() - start_time <self.setting_workspace_time:
                rate.sleep()
            else:
                break
        rospy.loginfo("Workspace published.")


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
    move_group.set_planning_time(self.max_planning_time)  # Increase this value to give more time for planning

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
            pose_goal.pose.position.z = 1.6
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
        # move_to_predefined_pose()
        # exit(0)
        
        jaco_arm_control = Jaco2ControlClient(standalone=True)
        while True:
            # Prompt user for input
            user_input = input("Enter 7 float numbers for quaternion (x, y, z, w) separated by spaces, or type 'quit' to exit: ")
        
            if user_input.lower() == "quit":
                rospy.loginfo("Quitting the program...")
                moveit_commander.roscpp_shutdown()
                sys.exit(0)

            # pose = np.array([0.2, 0.2 , 0.6, 0,0,0,1])
            
            # Convert input to a numpy array
            pose = np.array([float(num) for num in user_input.split()])
            # jaco_arm_control.move_arm_to_pose(pose)
            pose = np.array([0.30458, -0.3514, 0.46691, 0.10512, -0.68457, -0.12612, 0.71022])
            jaco_arm_control.check_plan_exists(pose)
        
        
    except rospy.ROSInterruptException:
        pass
