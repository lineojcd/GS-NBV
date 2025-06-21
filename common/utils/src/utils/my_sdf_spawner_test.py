import rospy
import rospkg
import numpy as np

from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import os



def load_sdf_model(model_name, model_path, model_pose, reference_frame="world"):
    # Read the SDF file
    with open(model_path, "r") as sdf_file:
        sdf_xml = sdf_file.read()
    
    # Call the service to spawn the model
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp = spawn_sdf(model_name, sdf_xml, "/", model_pose, reference_frame)
        rospy.loginfo("Model spawned: %s", resp.status_message)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def load_models(model_name_list, model_path_list, initial_pose_list, reference_frame="world"):
    for obj in zip(model_name_list, model_path_list, initial_pose_list):
        (model_name, model_path, initial_pose) = obj
        
        # Convert initial_pose to a Pose object
        pose = Pose()
        pose.position.x = initial_pose[0]
        pose.position.y = initial_pose[1]
        pose.position.z = initial_pose[2]
        pose.orientation.x = initial_pose[3]
        pose.orientation.y = initial_pose[4]
        pose.orientation.z = initial_pose[5]
        pose.orientation.w = 1.0  # Assuming no rotation in quaternion w component
        model_path = os.path.expandvars(model_path)

        load_sdf_model(model_name, model_path, pose)

def delete_tree_fruit():
        rospy.wait_for_service("gazebo/delete_model")
        delete_model_service = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        try:
            delete_model_service(model_name="mytree_100")
            delete_model_service(model_name="myavo_50")
            delete_model_service(model_name="myavo_50_a")
            delete_model_service(model_name="newavo_30")
        except Exception as e:
            print("delete box failed")




if __name__ == '__main__':
    rospy.init_node('spawn_sdf_model')
    
    # SDFSpawner().delete_bunny()
    
    # model_name_list =["mytree_100","myavo_50","myavo_50_a" ,"newavo_30"]
    # model_name_list =["Sketchfab Avocado Tree","myavo_50_red"]
    model_name_list =["Sketchfab Avocado Tree","myavo_50"]
    # model_name_list =["myavo_50_red"]
    # model_name_list =["myavo_50"]
    # model_name_list =["Kami_Rapacz_avo_straight_thick"]
    # model_name_list =["myavo_50"]
    model_path_list = [
                        # "/home/jcd/catkin_ws/src/vpp_avocado/sdf/mytree_100/model.sdf",
                       "/home/jcd/.gazebo/models/avocado_tree_sketchfab/model.sdf",
                       "/home/jcd/catkin_ws/src/vpp_avocado/sdf/myavo_50/model.sdf"
                    #    "/home/jcd/catkin_ws/src/vpp_avocado/sdf/myavo_50_red/model.sdf"
                    #    ,
                    #    "/home/jcd/catkin_ws/src/vpp_avocado/sdf/myavo_50_red/model.sdf",
                    #    ,
                    #    "/home/jcd/catkin_ws/src/vpp_avocado/sdf/Kami_Rapacz_avo_straight_thick/model.sdf"
                    #    ,
                    #    "/home/jcd/catkin_ws/src/vpp_avocado/sdf/myavo_50/model.sdf",
                    #    "/home/jcd/catkin_ws/src/vpp_avocado/sdf/newavo_30/model.sdf"
                       ]
    
    # init_position = [0,0,0]
    
    # init_posi = [1.05,-0.1,0.77]
    # init_posi = [0, 0 , 0.20]
    init_posi = [0.9, 0.15 , 0.70]
    
    initial_pose_list = [  # [x, y, z, roll, pitch, yaw]
                            rospy.get_param("~initial_pose", [-0.4+init_posi[0], -0.55+init_posi[1], -0.04 +init_posi[2], 0, 0, 0.46]) ,
                            
                            rospy.get_param("~initial_pose", [-0.4+init_posi[0], -0.41+init_posi[1], 0.46+init_posi[2], 0, 0, 0]) 
                            # ,
                            # rospy.get_param("~initial_pose", [-0.45+init_posi[0], -0.56+init_posi[1], 0.385+init_posi[2], 0, 0, 0]) ,
                            # rospy.get_param("~initial_pose", [-0.438+init_posi[0], -0.490+init_posi[1], 0.4803+init_posi[2], 0, 0, 0])
                        ]
    
    # delete_tree_fruit()
    load_models(model_name_list, model_path_list, initial_pose_list)    
    
    