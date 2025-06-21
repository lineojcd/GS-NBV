import rospy
import numpy as np

import time
import torch
from viewpoint_planners.my_jaco_control_client import Jaco2ControlClient
from perception.my_perceiver import MyPerceiver
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation
from utils.my_sdf_spawner import MySDFSpawner
from utils.sdf_spawner import SDFSpawner
from utils.py_utils import numpy_to_pose
from utils.py_utils import look_at_rotation, get_arm_pose, my_pause_program
import tf2_ros
from scipy.spatial.transform import Rotation as R

from viewpoint_planners.mygradientnbv_planner import GradientNBVPlanner


class GradientNBVPlanning:
    def __init__(self):
        self.model_init_pose = [float(x) for x in rospy.get_param('model_init_pose').split()]
        self.init_tar_posi = [float(x) for x in rospy.get_param('init_tar_posi').split()]
        self.picking_threshold = rospy.get_param("picking_threshold", "0.8")
        self.init_cam_dist = rospy.get_param("init_cam_dist", "0.40")
        self.avo_height_offset = rospy.get_param("avo_height_offset") 
        self.planning_flag = True
        self.picking_measure = None                         #(picking_condi, occlusion_rate)
        self.fruit_posi_GT = [float(x) for x in rospy.get_param('fruit_posi_GT').split()] 
        self.max_planning_iters = rospy.get_param("max_planning_iters", "10")
        self.planning_iters = 0
        self.perceiver = MyPerceiver()
        self.sdf_spawner = MySDFSpawner(init_pose=self.model_init_pose) # Spawne: avocado and tree
        camera_info = self.perceiver.get_camera_info()
        self.image_size = np.array([camera_info.width, camera_info.height])  #[640, 640]
        self.intrinsics = np.array(camera_info.K).reshape(3, 3)
        self.target_frame = rospy.get_param('target_frame')             #"d435_link"
        self.source_frame = rospy.get_param('source_frame')             #"j2n6s300_end_effector"
        self.arm_type = rospy.get_param('arm_type')
        self.init_arm_setting_time = rospy.get_param("init_arm_setting_time", "5.0")
        self.arm_move_waitting_time = rospy.get_param("arm_move_waitting_time", "5.0")
        self.experiment_env = rospy.get_param('experiment_env')
        self.verbose_planning = rospy.get_param("verbose_planning")
        self.exp_group = [float(x) for x in rospy.get_param('exp_group').split()]
        self.arm_control = Jaco2ControlClient(obstacle_center=self.model_init_pose,standalone = False)
        self.config_init_pose()     # Set up prep: target pos, occlusion pos
        self.f_pos_update_rate = rospy.get_param("fruit_pose_update_rate", "0.70")    
        self.f_pos_last = None
        self.f_axis_last = None
        self.cam_pose_history = []                      #(pose, fruit_coverage)
        self.obstacle_points_list = []
        self.fruit_pose_history = []
        self.picking_measure_history = [] 
        # self.planning_start = 0
        self.set_camera_params = True
        
        # Gradient-based planner
        self.gradient_planner = GradientNBVPlanner(
            image_size = self.image_size, intrinsics = self.intrinsics,
            start_pose = self.optical_pose, arm_control = self.arm_control,
            # target_params=self.target_position
            # TODO: should actually give self.f_pos
            # target_params = self.fruit_posi_GT)
            )
        
    def get_cam_2_ee_transform(self, target_frame, source_frame):
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
        # group_1 = np.array([0,0,-0.025])
        # group_2 = np.array([0,0,0])
        # group_3 = np.array([-0.06,0,0])
        
        # group = group_1
        # group = group_2
        # group_1 = -0.025
        # # group_2 = 0
        # group = group_2
        # group_test = 0.06
        # group = self.exp_group
        print("self.exp_group")
        
        rospy.loginfo(f"self.exp_group: {self.exp_group}")
        posi = np.array([self.init_tar_posi[0]+self.exp_group[0], self.init_tar_posi[1] + self.init_cam_dist + self.exp_group[1], self.init_tar_posi[2] + self.exp_group[2]])
        orientation = my_look_at_rotation(posi, self.init_tar_posi, plot=False, verbose=self.verbose_planning)
        self.optical_pose = np.concatenate((posi, orientation))
        
        trans = self.get_cam_2_ee_transform(self.target_frame, self.source_frame)
        if self.experiment_env == "real":
            print("Handle cam_ee_translation in real experiments")
            self.cam_ee_translation = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        
        self.R_cam_ee = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
        self.R_o_cam = np.array([float(x) for x in rospy.get_param('rot_optical_to_cam').split()])
        self.arm_pose = get_arm_pose(posi, orientation, self.R_cam_ee, self.R_o_cam)    # Quaternion form
        rospy.loginfo(f"Configure init arm pose: {self.arm_pose}, optical pose: {self.optical_pose}")
        rospy.loginfo(f"self.exp_group: {self.exp_group}")
        # print("<><>><<><><><><><")
        print("self.exp_group", self.exp_group)
        is_success = self.arm_control.move_arm_to_pose(self.arm_pose)
        rospy.sleep(self.init_arm_setting_time) 
        if not is_success:
            print("Arm is not working! Exit!")
            exit(0)
            
        # my_pause_program()
        self.planning_start = time.time()
        print("planning_start",self.planning_start)

    def run(self):
        self.run_gradient_nbv()
    
    def show_exp_sumamry(self):
        self.planning_duration = time.time() - self.planning_start
        rospy.loginfo(f"The planning spent {self.planning_duration} seconds")
        print("Show fruit_pose_history:")
        if np.array_equal(self.fruit_pose_history[-1][-1], np.array([0,0,1])):
            print(self.fruit_pose_history)
        else:
            print(self.fruit_pose_history[-1])
        print("Show picking_measure_history:")
        print(self.picking_measure_history)
        print("Show planning iters:")
        print(self.planning_iters)
        print("Show No of raycasting:")
        print(self.gradient_planner.get_num_raycasting())
        print("Show cam pose and ROI coverage history:")
        print(self.cam_pose_history)        # self.cam_pose_history.append((self.optical_pose, self.roi_coverage))
    
    def get_picking_score(self):
        picking_condi, occlusion_rate = self.picking_measure
        picking_score = picking_condi * (1 - occlusion_rate)
        return picking_score

    def run_gradient_nbv(self):
        if self.planning_iters <= self.max_planning_iters:
            self.sensing()
            self.update_semantic_octomap()
            self.visualize_octomap()
            picking_score = self.get_picking_score()

            if picking_score >= self.picking_threshold:
                self.show_exp_sumamry()
                rospy.loginfo(f"Fruit is pickable in the current view! Whole planning take {self.planning_duration} second")
                exit(0) 
            else:
                if self.planning_iters != self.max_planning_iters :
                    self.nbv_planning()
                    self.show_exp_sumamry()
                else:
                # if self.planning_iters == self.max_planning_iters :
                    self.show_exp_sumamry()
                    exit(0) 
        else:
            self.show_exp_sumamry()
            my_pause_program()
            exit(0) 
            
    def update_fruit_pose(self, f_pos, f_axis):
        if self.f_axis_last is not None:
            self.f_pos = self.f_pos_last + self.f_pos_update_rate * (f_pos - self.f_pos_last)
            self.f_axis = self.f_axis_last + self.f_pos_update_rate * (f_axis - self.f_axis_last)
        else:
            self.f_pos = f_pos
            self.f_axis = f_axis
        self.fruit_pose_history.append([self.f_pos, self.f_axis])
        
        if self.verbose_planning:
            print("type of current iter f_pos", type(f_pos))   # ndarray
            print("type of fruit_posi_GT", type(self.fruit_posi_GT))             # list
            print("GT fruit position:", self.fruit_posi_GT)
        print("Last iter fruit position and axis are:", self.f_pos_last, self.f_axis_last)
        print("current iter Est. fruit position and axis are:", f_pos, f_axis)
        print("current iter Est. error of fruit position:", f_pos - self.fruit_posi_GT)
        print("weighted current iter Est. fruit position and axis are:", self.f_pos, self.f_axis)
        print("weighted current iter Est. error of fruit position:", self.f_pos - self.fruit_posi_GT)
        
        self.f_pos_last = self.f_pos.copy()
        self.f_axis_last = self.f_axis.copy()
    
        
    def sensing(self):
        self.depth_img, self.keypoint_dict,self.semantics, self.picking_measure = self.perceiver.run() 
        # if (self.picking_measure[0]==False) and (self.picking_measure[1] ==1):
        if (self.picking_measure[0]==False) and (self.picking_measure[1] ==1):
            self.picking_measure = (True, 0.01)
        self.picking_measure_history.append(self.picking_measure)
        
        if self.optical_pose is not None:
            self.f_pos, self.f_axis = self.perceiver.estimate_fruit_pose(self.depth_img,self.keypoint_dict,self.optical_pose)  
            # f_pos, f_axis = self.perceiver.estimate_fruit_pose(self.depth_img,self.keypoint_dict,self.optical_pose)  
        self.fruit_pose_history.append([self.f_pos, self.f_axis])
        rospy.logwarn("Not using weighted fruit pose and axis here")
        # GS_NBV update rate
        # self.update_fruit_pose(f_pos, f_axis)
        print("f_pos by GS-NBV is:", self.f_pos)
    
    def update_semantic_octomap(self):
        rospy.loginfo(f"3D Representation: Building Semantic Octomap")
        start_time = time.time()
        if self.optical_pose is not None:
            self.roi_coverage = self.gradient_planner.update_semantic_octomap(self.depth_img, self.semantics, self.optical_pose)
            print("Iter {} fruit coverage {:.2f}% Building Octomap takes:{} sec".format(self.planning_iters, self.roi_coverage * 100, round(time.time() - start_time, 4)))
        self.cam_pose_history.append((self.optical_pose, self.roi_coverage))

    def visualize_octomap(self):
        self.gradient_planner.my_visualize(self.f_pos.flatten(), self.f_axis)
    
    def nbv_planning(self):
        rospy.loginfo(f"Planning for the next best view")
        start_time = time.time()
        
        # target_params = self.fruit_posi_GT
        if self.set_camera_params:
            print("self.f_pos  Should be nparray")
            self.gradient_planner.set_camera_params(self.f_pos)
            self.set_camera_params = False
        # self.optical_pose, loss, iters = self.gradient_planner.next_best_view()
        self.optical_pose, loss, iters = self.gradient_planner.next_best_view(self.f_pos)
        print("result of nbv from Gnbv planner:",self.optical_pose, loss, iters)
        self.planning_iters += 1
        print(f"View Planning process takes {round(time.time() - start_time, 4)} seconds")
        if self.optical_pose is not None:
            self.arm_pose = get_arm_pose(self.optical_pose[:3], self.optical_pose[3:], self.R_cam_ee, self.R_o_cam) 
            
            
            # Just for test
            print(" Next arm pose is : ",self.arm_pose)
            # my_pause_program()
            
            arm_is_success = self.arm_control.move_arm_to_pose(self.arm_pose)
            rospy.sleep(self.arm_move_waitting_time) 
            if arm_is_success:
                rospy.loginfo(f"Camera Moved to the next pose: {self.optical_pose} ")
                # self.cam_pose_history.append((self.optical_pose, self.roi_coverage))
            else:
                print("the pose is out of the arm range")
                exit(0)