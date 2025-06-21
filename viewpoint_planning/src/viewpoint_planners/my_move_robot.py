#! /usr/bin/env python3
"""Publishes joint trajectory to move robot to given pose"""

import rospy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from std_srvs.srv import Empty
import argparse
import time

class Jaco2ControlClient:
    """
    Gets data from the camera and performs semantic segmentation and pose estimation.
    """

    def __init__(self):
        self.prefix='j2n6s300'
        self.nbJoints = 6
        self.nbfingers = 3
        
    def moveJoint (self, pose):
      topic_name = '/' + self.prefix + '/effort_joint_trajectory_controller/command'
      pub = rospy.Publisher(topic_name, JointTrajectory, queue_size=1)
      jointCmd = JointTrajectory()  
      point = JointTrajectoryPoint()
      jointCmd.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.0);  
      point.time_from_start = rospy.Duration.from_sec(5.0)
      for i in range(0, self.nbJoints):
        jointCmd.joint_names.append(self.prefix +'_joint_'+str(i+1))
        point.positions.append(pose[i])
        point.velocities.append(0)
        point.accelerations.append(0)
        point.effort.append(0) 
      jointCmd.points.append(point)
      rate = rospy.Rate(100)
      count = 0
      while (count < 50):
        pub.publish(jointCmd)
        count = count + 1
        rate.sleep()     

    def moveFingers (self, jointcmds):
      topic_name = '/' + self.prefix + '/effort_finger_trajectory_controller/command'
      pub = rospy.Publisher(topic_name, JointTrajectory, queue_size=1)  
      jointCmd = JointTrajectory()  
      point = JointTrajectoryPoint()
      jointCmd.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.0);  
      point.time_from_start = rospy.Duration.from_sec(5.0)
      for i in range(0, self.nbJoints):
        jointCmd.joint_names.append(self.prefix +'_joint_finger_'+str(i+1))
        point.positions.append(jointcmds[i])
        point.velocities.append(0)
        point.accelerations.append(0)
        point.effort.append(0) 
      jointCmd.points.append(point)
      rate = rospy.Rate(100)
      count = 0
      while (count < 500):
        pub.publish(jointCmd)
        count = count + 1
        rate.sleep()     

if __name__ == '__main__':
  rospy.init_node('move_robot_using_trajectory_msg')	
  arm = Jaco2ControlClient()	
  home_pose = [0.0,2.9,1.3,4.2,1.4,0.0]
  pose = [1.0,2.9,1.3,4.2,1.4,0.0]
  res = arm.moveJoint(home_pose) 
  time.sleep(4)
  print("back in home position now, joints are: [0.0,2.9,1.3,4.2,1.4,0.0]")
  print("res = ",res)
  res = arm.moveJoint(pose) 
  time.sleep(4)
  res = arm.moveJoint(home_pose) 
  time.sleep(4)
  print("back in home position now, joints are: [0.0,2.9,1.3,4.2,1.4,0.0]")
  
  # Unpause the physics
  # rospy.wait_for_service('/gazebo/unpause_physics')
  # unpause_gazebo = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
  # resp = unpause_gazebo()

      

