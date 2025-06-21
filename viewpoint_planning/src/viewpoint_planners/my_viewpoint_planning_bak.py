import rospy
import numpy as np

import time
import torch
from abb_control.arm_control_client import ArmControlClient
from viewpoint_planners.my_jaco_control_client import Jaco2ControlClient
from perception.my_perceiver import MyPerceiver
from perception.perceiver import Perceiver
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from viewpoint_planners.my_viewpoint_sampler import MyViewpointSampler
from viewpoint_planners.gradientnbv_planner import GradientNBVPlanner
from viewpoint_planners.my_nbv_planner import MyNBVPlanner
from viewpoint_planners.random_planner import RandomPlanner
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation
from utils.my_sdf_spawner import MySDFSpawner
from utils.sdf_spawner import SDFSpawner
from utils.py_utils import numpy_to_pose
from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import look_at_rotation, get_arm_pose, my_pause_program
import tf2_ros
from scipy.spatial.transform import Rotation as R
import numpy as np

  

class MyViewpointPlanning:
    def __init__(self):
        self.model_init_pose = [float(x) for x in rospy.get_param('model_init_pose').split()]
        self.init_tree_posi = [float(x) for x in rospy.get_param('init_tree_posi').split()]
        self.picking_threshold = rospy.get_param("picking_threshold", "0.8")
        self.init_cam_dist = rospy.get_param("init_cam_dist", "0.55")    
        self.avo_height_offset = rospy.get_param("avo_height_offset") 
        self.planning_flag = True
        self.picking_score = {}                         #key: score, value: campera pose
        self.cam_pose_history = []                      #(pose, fruit_coverage)
        self.fruit_posi_GT = [float(x) for x in rospy.get_param('fruit_posi_GT').split()] 
        self.planning_iters = 0
        self.perceiver = MyPerceiver()
        self.sdf_spawner = MySDFSpawner(init_pose=self.model_init_pose) # Spawne: avocado and tree
        camera_info = self.perceiver.get_camera_info()
        self.image_size = np.array([camera_info.width, camera_info.height])  #[640, 640]
        self.intrinsics = np.array(camera_info.K).reshape(3, 3)
        self.target_frame = rospy.get_param('target_frame')             #"d435_link"
        self.source_frame = rospy.get_param('source_frame')             #"j2n6s300_end_effector"
        self.arm_type = rospy.get_param('arm_type')
        self.experiment_env = rospy.get_param('experiment_env')
        self.verbose_planning = rospy.get_param("verbose_planning")
        self.arm_control = Jaco2ControlClient(obstacle_center=self.model_init_pose,standalone = False)
        self.config_init_pose()     # Set up prep: target pos, occlusion pos
        self.mynbv_planner = MyNBVPlanner(self.image_size, self.intrinsics)
        
    def get_cam_2_ee_transform(self, target_frame, source_frame):
        # rospy.init_node('tf_listener_node')
        # Create a TF2 buffer and listener
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        try:
            # Wait for up to 5 seconds for the transform
            trans = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(5.0))
            rospy.loginfo("Transform from {} to {}: \n{}".format(target_frame,source_frame, trans.transform.rotation))
            return trans
        except tf2_ros.LookupException as e:
            rospy.logerr("Transform not found: {}".format(e))
            return None
        except tf2_ros.ConnectivityException as e:
            rospy.logerr("Transform connectivity error: {}".format(e))
            return None
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr("Transform extrapolation error: {}".format(e))
            return None
    
    def config_init_pose(self):
        # Set initial tree position for the ENV: pre-defined distance, based on init target pose
        posi = np.array([self.init_tree_posi[0], self.init_tree_posi[1] + self.init_cam_dist, self.init_tree_posi[2]])
        orientation = look_at_rotation(posi, self.init_tree_posi)
        print("use look_at_rotation: ",orientation)
        orientation = my_look_at_rotation(posi, self.init_tree_posi, plot=False, verbose=self.verbose_planning)
        print("use my_look_at_rotation: ",orientation)
        
        self.optical_pose = np.concatenate((posi, orientation))
        
        trans = self.get_cam_2_ee_transform(self.target_frame, self.source_frame)
        if self.experiment_env == "real":
            print("Handle cam_ee_translation in real experiments")
            self.cam_ee_translation = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        
        self.R_cam_ee = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
        self.R_o_cam = np.array([float(x) for x in rospy.get_param('rot_optical_to_cam').split()])
        self.arm_pose = get_arm_pose(posi, orientation, self.R_cam_ee, self.R_o_cam)    # Quaternion form
        rospy.loginfo(f"Configure init arm pose: {self.arm_pose}, optical pose: {self.optical_pose}")
        
        is_success = self.arm_control.move_arm_to_pose(self.arm_pose)
        rospy.sleep(2.0) 
        if not is_success:
            print("Arm is not working! Exit!")
            exit(0)
        
    def run(self):
        self.run_mynbv()
        
    def run_mynbv(self):
        if self.planning_flag:
            # TODO: maybe needed later
            # self.mynbv_planner.update_info()
            
            rospy.loginfo(f"Sensing: Estimate fruit position and axis in the current view")
            self.depth_img, self.keypoint_dict,self.semantics, self.picking_score = self.perceiver.run() 
            self.fruit_position, self.fruit_axis = self.perceiver.estimate_fruit_pose(self.depth_img,self.keypoint_dict,self.optical_pose)  
            self.obstacle_points = self.perceiver.estimate_obstacle_points(self.depth_img,self.keypoint_dict,self.optical_pose)
            
            if self.verbose_planning:
                print("GT fruit position:", self.fruit_posi_GT)
                print("Est. error of fruit position:", self.fruit_position - self.fruit_posi_GT)
            print("Est. fruit position and axis are:", self.fruit_position, self.fruit_axis)
            print("self.obstacle_points:",self.obstacle_points)
            rospy.loginfo(f"3D Representation: Building Semantic Octomap")
            start_time = time.time()
            fruit_coverage = self.mynbv_planner.update_semantic_octomap(self.depth_img, self.semantics, self.optical_pose)
            print("Iter {} fruit coverage {:.2f}% Building Octomap takes:{} sec".format(self.planning_iters, fruit_coverage * 100,  
                    round(time.time() - start_time, 4),))
            self.mynbv_planner.my_visualize(self.fruit_position.flatten(), self.fruit_axis )   
            
            # Staging the fruit
            if self.picking_score >= self.picking_threshold:
                staging_optical_pose = self.mynbv_planner.get_staging_pose(self.fruit_position)
                self.arm_pose = get_arm_pose(staging_optical_pose[:3], staging_optical_pose[3:], self.R_cam_ee, self.R_o_cam) 
                rospy.loginfo(f"Fruit is pickable in the current view! Move to staging pose: {staging_optical_pose}")
                self.arm_control.remove_all_obstacles()
                arm_is_success = self.arm_control.move_arm_to_pose(self.arm_pose)
                if arm_is_success:
                    self.planning_flag = False
                    rospy.loginfo(f"The robot has reached the staging point. Terminating the program ... ")
                    exit(0) 
            else:
                rospy.loginfo(f"Planning for the next best view")
                start_time = time.time()
                self.optical_pose = self.mynbv_planner.my_next_best_view(self.fruit_position,self.fruit_axis, self.obstacle_points, self.optical_pose)
                print(f"View Planning process takes {round(time.time() - start_time, 4)} seconds")
                self.arm_pose = get_arm_pose(self.optical_pose[:3], self.optical_pose[3:], self.R_cam_ee, self.R_o_cam) 
                arm_is_success = self.arm_control.move_arm_to_pose(self.arm_pose)
                
                my_pause_program()                 # For my test
                
                
                if arm_is_success:
                    rospy.loginfo(f"Camera Moved to the next pose: {self.optical_pose} ")
                    self.cam_pose_history.append((self.optical_pose, fruit_coverage))
                else:
                    print("the pose is out of the arm range")
                    exit(0)
                rospy.sleep(1.0)       
                
                

if __name__ == '__main__':
    # My test tensor
    arm_control = ArmControlClient()
    my_viewpoint_sampler = MyViewpointSampler()
    model_init_pose= [0.9, 0.15 , 0.70]
    sdf_spawner = MySDFSpawner(init_pose = model_init_pose)
    
    init_target_distance = 0.35
    fruit_sample_radius = 0.27
    fruit_sample_radius = 0.27
    init_target_align_position = np.array([0.5, -0.4, 1.18])
    # target_position = init_target_align_position + np.array([0, 0, 0.029])
    camera_pose = my_viewpoint_sampler.predefine_start_pose(init_target_align_position, init_target_distance)
    print("Initi camera_pose is ",camera_pose)
    # camera_pose = np.array([0.5,-0.05 , 1.18, 0.70711 ,0,0,-0.70711])
    arm_is_success = arm_control.move_arm_to_pose(numpy_to_pose(camera_pose))
    print("arm_is_success: ", arm_is_success)
    # Sleep for 3 seconds
    time.sleep(3)

    # if False: 
    if True: 
        # new_camera_pose = np.array([0.51+ 0.3 ,-0.2, 1.16, 0.70711 ,0,0,-0.70711])
        fruit_position = np.array([0.50, -0.26, 1.16+0.029 ])
        # new_camera_position = np.array([0.5 + fruit_sample_radius ,-0.26, 1.16+0.029])
        new_camera_position = np.array([0.5 - fruit_sample_radius ,-0.26, 1.16+0.029])
        # new_camera_position = np.array([0.51- fruit_sample_radius ,-0.2, 1.16])
        orientation = look_at_rotation(new_camera_position, fruit_position)
        new_camera_pose = np.concatenate((new_camera_position, orientation))
        print("new_camera_pose is : ", new_camera_pose)
        arm_is_success = arm_control.move_arm_to_pose(numpy_to_pose(new_camera_pose))
        print("arm_is_success: ", arm_is_success)
    else:
        print("the pose is out of the arm range")
    
    