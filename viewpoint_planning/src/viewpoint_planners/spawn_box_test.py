import rospy
import rospkg
import numpy as np

from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import os

box_sdf = """
<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='box'>
    <static>false</static>
    <link name='link'>
      <gravity>false</gravity> <!-- Disable gravity for the box -->
        <inertial>
        <mass>0.000</mass>
        <inertia>
          <ixx>0.00</ixx>
          <ixy>0.00</ixy>
          <ixz>0.00</ixz>
          <iyy>0.00</iyy>
          <iyz>0.00</iyz>
          <izz>0.00</izz>
        </inertia>
      </inertial>
      <collision name='collision'>
        <geometry>
          <box>
            <size>0.01 0.13 0.63</size> <!-- Box dimensions: width, height, depth -->
          </box>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <box>
            <size>0.01 0.13 0.63</size> <!-- Box dimensions: width, height, depth -->
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient> <!-- Box color: grey -->
          <diffuse>0.5 0.5 0.5 1</diffuse> <!-- Box color: grey -->
        </material>
      </visual>
      <pose>-0.08 -0.32 -0.2 0 0 -0.6</pose>
    </link>
  </model>
</sdf>
"""


def spawn_box():
    rospy.init_node('spawn_box_node', anonymous=True)
    
    # Wait for the service to become available
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    
    try:
        # Define the service proxy
        spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        # Define the box's initial pose
        box_pose = Pose(Point(0, 0, 0.5), Quaternion(0, 0, 0, 1))  # Set the position and orientation
        
        # Call the service to spawn the box
        spawn_model_prox("box", box_sdf, "", box_pose, "world")
        
        rospy.loginfo("Box spawned in Gazebo!")
        
    except rospy.ServiceException as e:
        rospy.logerr("Failed to spawn box: %s" % e)

def spawn_fruit():
    rospy.init_node('spawn_fruit', anonymous=True)
    
    # Wait for the service to become available
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    
    try:
        # Define the service proxy
        spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        model_name = "myavo_50_tilt_left"
        model_path = "/home/jcd/catkin_ws/src/vpp_avocado/sdf/"+model_name+"/model.sdf"
        # Define the box's initial pose
        # initial_pose_list = [  # [x, y, z, roll, pitch, yaw]
        #     rospy.get_param("~initial_pose", [init_posi[0], 0.14+init_posi[1], 0.50+init_posi[2], 0, 0, 1.856792]) ]
        model_pose = Pose(Point(0.5, 0, 0.5), Quaternion(0, 0, 0, 1))  # Set the position and orientation

        # Read the SDF file
        with open(model_path, "r") as sdf_file:
            sdf_xml = sdf_file.read()
        
        # Call the service to spawn the model
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            resp = spawn_sdf("test_fruit1", sdf_xml, "/", model_pose, "world")
            rospy.loginfo("Model spawned: %s", resp.status_message)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
        
        rospy.loginfo("Box spawned in Gazebo!")
        
    except rospy.ServiceException as e:
        rospy.logerr("Failed to spawn box: %s" % e)

if __name__ == '__main__':
    try:
        spawn_box()
        # spawn_fruit()
    except rospy.ROSInterruptException:
        pass
