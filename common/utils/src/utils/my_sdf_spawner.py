import rospy
import rospkg
import numpy as np

from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import os

class MySDFSpawner:
    def __init__(self, model_name="box", init_pose= [0.9, 0.15 , 0.70]):
        
        # ROS service for spawning.
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        
        # define using real avocado model or fake model for toy example
        # self.fruit_model = "real"  # or red for toy_example
        # self.fruit_model = "red"  
        self.fruit_model = rospy.get_param('fruit_model')
        self.workspace_path = rospy.get_param('workspace_path')
        self.box_sdf = """
                <?xml version='1.0'?>
                <sdf version='1.6'>
                <model name='box'>
                    <static>false</static>
                    <link name='link'>
                     <gravity>false</gravity> <!-- Disable gravity for the box -->
                    <inertial>
                    <mass>0.001</mass>
                    <inertia>
                        <ixx>0.01</ixx>
                        <ixy>0.01</ixy>
                        <ixz>0.01</ixz>
                        <iyy>0.01</iyy>
                        <iyz>0.01</iyz>
                        <izz>0.01</izz>
                    </inertia>
                    </inertial>           
                    <collision name='collision'>
                        <geometry>
                        <box>
                            <size>0.01 0.13 0.6</size> <!-- Box dimensions: width, height, depth -->
                        </box>
                        </geometry>
                    </collision>
                    <visual name='visual'>
                        <geometry>
                        <box>
                            <size>0.01 0.13 0.6</size> <!-- Box dimensions: width, height, depth -->
                        </box>
                        </geometry>
                        <material>
                        <ambient>0.5 0.5 0.5 1</ambient> <!-- Box color: grey -->
                        <diffuse>0.5 0.5 0.5 1</diffuse> <!-- Box color: grey -->
                        </material>
                    </visual>
                    <pose>-0.1 -0.32 0 0 0 -0.6</pose>
                    </link>
                </model>
                </sdf>
                """
        
        
        self.init_posi = init_pose
        
        self.model_name = model_name
        rospack = rospkg.RosPack()
        self.model_path = rospack.get_path("simulation_environment") + "/sdfs/"
        
        # Test case time
        # self.delete_bunny()
        # self.delete_box()
        self.my_test_model()
        # self.add_mybox()

    def add_mybox(self):
        # Wait for the service to become available
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            # Define the service proxy
            spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            # Define the box's initial pose
            box_pose = Pose(Point(0, 0, 0.3), Quaternion(0, 0, 0, 1))  # Set the position and orientation
            # Call the service to spawn the box
            spawn_model_prox("box", self.box_sdf, "", box_pose, "world")
            rospy.loginfo("Box spawned in Gazebo!")
        except rospy.ServiceException as e:
            rospy.logerr("Failed to spawn box: %s" % e)        
    
    def spawn_box(self, pos):
        pos = pos - np.array([0.0, 0.0, 0.024])  # Correcting for Gazebo coordinates
        box_pose = Pose(
            position=Point(*pos),
            orientation=Quaternion(*quaternion_from_euler(0, 0, 0)),
        )
        try:
            spawner = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            spawner(
                model_name="box",
                model_xml=open(
                    self.model_path + "box.sdf",
                    "r",
                ).read(),
                robot_namespace="",
                initial_pose=box_pose,
                reference_frame="world",
            )
        except rospy.ServiceException as e:
            print("Service call failed: ", e)
            
    
    def spawn_avocado_tree(self, pos):
        #  can do sth here
        
        pass

    def delete_box(self):
        rospy.wait_for_service("gazebo/delete_model")
        delete_model_service = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        try:
            delete_model_service(model_name="box")
        except Exception as e:
            print("delete box failed")
    
    def delete_bunny(self):
        rospy.wait_for_service("gazebo/delete_model")
        delete_model_service = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        try:
            delete_model_service(model_name="bunny")
        except Exception as e:
            print("delete box failed")


    def load_sdf_model(self,model_name, model_path, model_pose, reference_frame="world"):
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

    def load_models(self,model_name_list, model_path_list, initial_pose_list, reference_frame="world"):
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

            self.load_sdf_model(model_name, model_path, pose)

    def delete_tree_fruit(self):
            rospy.wait_for_service("gazebo/delete_model")
            delete_model_service = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
            try:
                delete_model_service(model_name="mytree_100")
                delete_model_service(model_name="myavo_50")
                delete_model_service(model_name="myavo_50_a")
                delete_model_service(model_name="newavo_30")
            except Exception as e:
                print("delete box failed")


    def my_test_model(self):
        if self.fruit_model == "real":
            model_name_list =["Sketchfab Avocado Tree","myavo_50"]
            # model_name_list =["Sketchfab Avocado Tree","myavo_50_tilt_right"]
            # model_name_list =["Sketchfab Avocado Tree","myavo_50_tilt_left"]
        else:
            model_name_list =["Sketchfab Avocado Tree","myavo_50_red"]
        model_folder_name  = model_name_list[1]
        model_path_list = [
                            # "/home/jcd/.gazebo/models/avocado_tree_sketchfab/model.sdf",
                            self.workspace_path + "simulation_environment/sdfs/"+"avocado_tree_sketchfab/model.sdf",
                            # "/home/jcd/catkin_ws/src/vpp_avocado/sdf/"+model_folder_name+"/model.sdf"
                            
                            # this is the path for PC4090
                            # "/home/jcd/gsnbv_ws/src/simulation_environment/sdfs/"+model_folder_name+"/model.sdf"
                            self.workspace_path + "simulation_environment/sdfs/"+model_folder_name+"/model.sdf"
                        #    "/home/jcd/catkin_ws/src/vpp_avocado/sdf/myavo_50_red/model.sdf",
                        ]
        # init_posi = [0, 0 , 0.20]
        # init_posi = [0.9, 0.1 , 0.70]
        # init_posi = [0.9, 0.15 , 0.70]
        init_posi = self.init_posi
        
        # print("I am sss s s s s schianging : <><><>><><><><><><><><><><>")
        
        initial_pose_list = [  # [x, y, z, roll, pitch, yaw]
                                # rospy.get_param("~initial_pose", [-0.4+init_posi[0], -0.55+init_posi[1], 0+init_posi[2], 0, 0, 0.26]) ,
                                rospy.get_param("~initial_pose", [init_posi[0], init_posi[1], init_posi[2], 0, 0, 0.46]) ,
                                # rospy.get_param("~initial_pose", [init_posi[0], 0.14+init_posi[1], 0.50+init_posi[2], 0, 0, 0]) 
                                rospy.get_param("~initial_pose", [init_posi[0], 0.16+init_posi[1], 0.50+init_posi[2], 0, 0, 1.856792]) 
                                # rospy.get_param("~initial_pose", [init_posi[0], 0.16+init_posi[1], 0.467+init_posi[2], 0, 0, 1.0]) 
                                # rospy.get_param("~initial_pose", [init_posi[0], 0.14+init_posi[1], 0.46+init_posi[2], 0, 0, 1.856792]) 
                            ]
        # self.delete_tree_fruit()
        self.load_models(model_name_list, model_path_list, initial_pose_list)
    


if __name__ == '__main__':
    rospy.init_node('spawn_sdf_model')
    
    # SDFSpawner().delete_bunny()
    init_pose = [0.0, -0.50, -0.04]
    
    model_spawner = MySDFSpawner(init_pose = init_pose)
    # model_spawner.my_test_model()
    
    
    
    