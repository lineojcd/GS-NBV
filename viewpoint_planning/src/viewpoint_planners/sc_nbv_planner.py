import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import rospy
import time
from math import exp
from scipy.spatial.transform import Rotation as R
from scene_representation.sc_voxel_grid import SCNBVVoxelGrid
from viewpoint_planners.sc_viewpoint_sampler import SCViewpointSampler
from viewpoint_planners.sc_rrt3d import RRT3D
from scene_representation.conversions import T_from_rot_trans_np, T_from_rot_trans
from utils.sc_rviz_visualizer import SCRvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array, look_at_rotation, my_pause_program
from utils.torch_utils import  transform_from_rotation_translation, transform_from_rot_trans, generate_pixel_coords, get_centroid, transform_points
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation

class SCNBVPlanner(nn.Module):
    """
    Class to plan a locally optimal viewpoint using gradient-based optimization
    """
    def __init__(self, image_size: np.array, intrinsics: np.array, arm_control) -> None:
        """
        :param image_size: size of the image in pixels width/height
        :param intrinsics: camera intrinsic
        """
        super(SCNBVPlanner, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.width=image_size[0]
        self.height=image_size[1]
        self.pick_offset = rospy.get_param("pick_offset", "0.2")           # In meters
        self.use_max_fusion = rospy.get_param("use_max_fusion")
        self.motion_plan_check = rospy.get_param("SCNBV_motion_plan_check")
        self.verbose_planner = rospy.get_param("verbose_planner")
        self.R_o_cam = np.array([float(x) for x in rospy.get_param('rot_optical_to_cam').split()])
        self.r_o_pc = np.array([float(x) for x in rospy.get_param('rot_optical_to_pixelcam').split()])
        self.vp_Gsem_filter = rospy.get_param("SCNBV_viewpoint_semantics_gain_filter", 0.0)                   
        self.vp_motion_cost_filter = rospy.get_param("SCNBV_viewpoint_motion_cost_filter") 
        self.search_space = rospy.get_param("SCNBV_search_space", "3D") 
        self.sc_viewpoint_sampler = SCViewpointSampler()
        self.rrt3d = RRT3D()
        self.rviz_visualizer = SCRvizVisualizer()
        self.voxel_grid = SCNBVVoxelGrid(self.width, self.height, intrinsics, self.device)
        self.arm_control = arm_control
        self.metric_score = 0
        self.metric_score_history = []
        self.vp_queue = []                  # existing vp in the queue
        self.vp_nbv_history = []            # moved nbv in history
        self.filtered_vp_score_history = []        # checked vp filtered out by IG or close to queue, nbv, previous checked vp
        # self.no_plan_vp_history= []       # No need for now
        self.num_raycasting = 0
        self.rrt_init = False
        self.map_limits_up = [float(x) for x in rospy.get_param('SCNBV_map_limits_up').split()]
        self.map_limits_bottom = [float(x) for x in rospy.get_param('SCNBV_map_limits_bottom').split()]
        self.map_limits = [self.map_limits_bottom, self.map_limits_up]
        self.obstacle_region_up = [float(x) for x in rospy.get_param('SCNBV_obstacle_region_up').split()]
        self.obstacle_region_bottom = [float(x) for x in rospy.get_param('SCNBV_obstacle_region_bottom').split()]
        self.obstacle_region = [self.obstacle_region_bottom, self.obstacle_region_up]
        
    def get_staging_pose_old(self, fruit_position ):
        staging_position = fruit_position + np.array([0,self.pick_offset,0])
        staging_position = staging_position.squeeze()       # Ensure pick_posi is 1D arrays
        fruit_position = fruit_position.squeeze()
        staging_orientation = my_look_at_rotation(staging_position.squeeze(), fruit_position.squeeze()) 
        staging_pose = np.concatenate((staging_position, staging_orientation))
        return staging_pose

    def get_staging_pose(self, f_pos, optical_pose):
        dir_vec =  optical_pose[:3] - f_pos 
        dir_vec /= np.linalg.norm(dir_vec)  # Normalize the vector
        dir_vec *= self.pick_offset
        staging_position = f_pos + dir_vec
        staging_position = staging_position.squeeze()       # Ensure pick_posi is 1D arrays
        f_pos = f_pos.squeeze()
        staging_orientation = my_look_at_rotation(staging_position, f_pos) 
        staging_pose = np.concatenate((staging_position, staging_orientation))
        return staging_pose
       
    def update_semantic_octomap(self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array) -> None:
        """
        Process depth and semantic images and insert them into the voxel grid
        :param depth_image: depth image (H x W)
        :param semantics: semantic confidence scores and class ids (H x W x 2)
        :param viewpoint: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(R.from_quat(viewpoint[3:]).as_matrix(), dtype=torch.float32, device=self.device)
        Trans_w_o = transform_from_rot_trans(orientation[None, :], position[None, :])    # get camera pose in 4*4 matrice
        if self.use_max_fusion:
            self.metric_score = self.voxel_grid.insert_depth_and_semantics_max_fusion(depth_image, semantics, Trans_w_o)
        return self.metric_score
    
    def viewpoints_sampling(self, Q_est, optical_pose):
        if self.search_space == "3D":
            valid_vp_list = self.sc_viewpoint_sampler.SCNBVP_sampler_3D_world(Q_est, optical_pose)
        if self.search_space == "2D":
            valid_vp_list = self.sc_viewpoint_sampler.SCNBVP_sampler_2D_world(Q_est, optical_pose)
        
        if self.verbose_planner:
            print("valid_vp_list is \n", valid_vp_list)
        return valid_vp_list

    def get_num_raycasting(self):
        return self.num_raycasting
    
    def viewpoints_evaluation(self, vp_candidates_list, optical_pose):
        cam_posi = torch.tensor(optical_pose[:3], dtype=torch.float32, device=self.device)
        cam_orien = torch.tensor(R.from_quat(optical_pose[3:]).as_matrix(), dtype=torch.float32, device=self.device)
        cam_Trans_w_o = transform_from_rot_trans(cam_orien[None, :], cam_posi[None, :])
        # cam_SC_score = self.voxel_grid.SC_viewpoint_evaluation(cam_Trans_w_o)       
        cam_SC_score = self.voxel_grid.SC_viewpoint_evaluation_new(cam_Trans_w_o)     
        self.num_raycasting += 1  
        print("Extract SC(M, cam): cam_SC_score is", cam_SC_score)
        vp_score_list = []
        
        # start_time = time.time()
        time_list = []
        
        for vp in vp_candidates_list:       # vp is an np array
            
            
            posi = torch.tensor(vp[:3], dtype=torch.float32, device=self.device)
            orien = torch.tensor(R.from_quat(vp[3:]).as_matrix(), dtype=torch.float32, device=self.device)
            vp_Trans_w_o = transform_from_rot_trans(orien[None, :], posi[None, :]) # get viewpose in 4*4 matrice
            
            
            
            # vp_SC_score = self.voxel_grid.SC_viewpoint_evaluation(vp_Trans_w_o)
            vp_SC_score = self.voxel_grid.SC_viewpoint_evaluation_new(vp_Trans_w_o)
            
            # milestone = time.time()
            # print(" this part take: ", milestone - start_time)
            # time_list.append( milestone - start_time)
            
            
            motion_cost = self.voxel_grid.compute_mostion_cost(posi, cam_posi).item()          # Compute e^(-lamda * d)
            SC_minimize = cam_SC_score - vp_SC_score
            vp_IG = SC_minimize * motion_cost
            
            
            # print("vp_SC_score:", vp_SC_score)
            # print("vp_IG, SC_minimize, motion_cost are: ",vp_IG, SC_minimize, motion_cost)
            rospy.loginfo(f"vp_IG, SC_minimize (cam_SC_score - vp_SC_score), motion_cost are: {vp_IG.item()}, {SC_minimize.item()} ({cam_SC_score.item()} - {vp_SC_score.item()}),{motion_cost}") 
            self.num_raycasting += 1
            
            vp_score_list.append((vp_IG, SC_minimize, motion_cost, vp))
        
        # print("time_list is ",time_list)
        # my_pause_program()    
        
        rospy.loginfo(f"NO Gsem/motion cost filter applied in SCNBV ... ")
        return vp_score_list
    
    def viewpoint_filtering(self, G_sem, motion_cost, posi):
        remove_viewpoint = False
        if self.vp_IGfilter_current_iteration(G_sem, motion_cost):
            return True
        return remove_viewpoint
    
    def vp_IGfilter_current_iteration(self, G_sem, motion_cost):
        remove_viewpoint = False
        if G_sem.item() <= self.vp_Gsem_filter:
            return True
        if motion_cost.item() <= exp(self.vp_motion_cost_filter):
            return True
        return remove_viewpoint
    
    def vp_distance_filter(self, posi, querylist):
        remove_viewpoint = False
        for item in querylist:
            vp_posi = item[-1][:3]
            distance = np.linalg.norm(vp_posi - posi)           # Calculate Euclidean distance
            if distance <= self.vp_motion_cost_filter:
                return True
        return remove_viewpoint
    
    def vp_filter_in_queue(self, posi):
        print("show me self.vp_queue: ", self.vp_queue)
        return self.vp_distance_filter(posi, self.vp_queue)
    
    def vp_filter_in_nbv_history(self, posi):
        return self.vp_distance_filter(posi, self.vp_nbv_history)
    
    def vp_filter_in_filtered_vp_history(self, posi):
        return self.vp_distance_filter(posi, self.filtered_vp_score_history)
    
    def sort_viewpoints_utility(self, vp_candidates_score_list):
        # Convert the score from tensor to float and sort the list by the score
        sorted_vp = sorted(vp_candidates_score_list, key=lambda x: x[0], reverse=True)
        for ig, pose in sorted_vp:
            print(f"Score: {round(ig, 2)}, Pose: {pose}")
        return sorted_vp

    def update_IG_in_queue(self, optical_pose):
        for idx, item in enumerate(self.vp_queue):
            vp_IG, G_sem, motion_cost, vp = item                    # Unpack the tuple
            vp_posi = vp[:3]                                        # Get the x, y, z coordinates (first 3 elements)
            distance = np.linalg.norm(vp_posi - optical_pose[:3])
            motion_cost = exp(distance)
            vp_IG = G_sem * motion_cost
            self.vp_queue[idx] = (vp_IG, G_sem, motion_cost, vp)    # Replace the motion with the distance 
    
    def motion_plan_exist_filter(self, vp_score_list):
        rospy.loginfo(f"check motion plan ...")
        reachable_vp_score_list = []
        for vp_score in vp_score_list:
            vp_pose = vp_score[-1]
            if self.check_trajectory_exist(vp_pose):
                reachable_vp_score_list.append(vp_score)  
            else:
                self.filtered_vp_score_history.append(vp_score)
        return reachable_vp_score_list
    
    def check_trajectory_exist(self, pose):
        return self.arm_control.check_trajectory_exist(pose)
    
    def viewpoint_queue_update(self, new_vp_IG_list):
        # rospy.logwarn("You might need to Build a search graph to contain the cam_poses and update it each time")
        rospy.loginfo(f"Merging current viewpoints to existing Queue")
        self.vp_queue = new_vp_IG_list
    
    def viewpoint_queue_sorting(self):
        self.vp_queue.sort(key=lambda x: x[0], reverse=True)
        for vp in self.vp_queue:
            print(f"Score: {vp[0]}, Pose: {vp[-1]}")
    
    def viewpoint_selection(self):
        nbv_vp_score = self.vp_queue.pop(0)
        return nbv_vp_score
    
    def convert_point_to_pose(self, vp_list, f_pos):
        sampled_vp_pose_list = []
        for posi in vp_list:
            # orie = my_look_at_rotation(posi, f_pos.squeeze().numpy())
            orie = my_look_at_rotation(posi, f_pos)
            if self.verbose_planner:
                print(" my_nbv_sampler my_look_at_rotation: ",orie)
            pose = np.concatenate((posi, orie))
            sampled_vp_pose_list.append(pose)
        return sampled_vp_pose_list
    
    def scnbv_position_to_pose(self, next_posi, Q_est):
        orie = my_look_at_rotation(next_posi, Q_est)
        if self.verbose_planner:
            print(" sc_nbv_sampler look_at_rotation: ",orie)
        pose = np.concatenate((next_posi, orie))
        return pose
        
    def sc_next_best_view(self, Q_est, optical_pose=None) -> np.array:
        """
        Compute the next best viewpoint
        :return: the NBV viewpoint of camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        rospy.loginfo(f"Sampling valid viewpoint candidates on a Spherical wedge/lune ...")
        
        # start_time = time.time()
        
        
        valid_vp_list = self.viewpoints_sampling(Q_est, optical_pose)
        
        
        
        # print("show me selected_vp_list", valid_vp_list) # np array: list of [x,y,z]
        self.rviz_visualizer.visualize_viewpointslist(valid_vp_list,Q_est, "valid_vps")
        sampled_vp_pose_list = self.convert_point_to_pose(valid_vp_list, Q_est)
        
        rospy.loginfo(f"Evaluating viewpoint candidates ... ")
        
        sampled_vp_pose_list = self.viewpoints_evaluation(sampled_vp_pose_list, optical_pose)
        
        if self.motion_plan_check:
            reachable_vp_score_list = self.motion_plan_exist_filter(sampled_vp_pose_list) # Check motion plan existing filter
            self.viewpoint_queue_update(reachable_vp_score_list)
        else:
            self.viewpoint_queue_update(sampled_vp_pose_list)
            
        self.viewpoint_queue_sorting()
        if len(self.vp_queue) ==0:
            return None
        else:
            
            
            
            nbv_vp_score = self.viewpoint_selection()  
            self.vp_nbv_history.append(nbv_vp_score)
            nbv_vp = nbv_vp_score[-1]
            self.rviz_visualizer.visualize_viewpointslist([nbv_vp],Q_est, "nbv_vp")
            
        
            
            next_posi = self.SC_RRT(optical_pose[:3], nbv_vp[:3], Q_est )
            next_vp = self.scnbv_position_to_pose(next_posi, Q_est)
            
            # print(" this part take: ", time.time() - start_time)
            # my_pause_program()
            
            return next_vp
    
    def SC_RRT(self,start, goal, Q_est):
        if not self.rrt_init:
            self.rrt3d.set_sampling_range(self.map_limits)
            self.rrt3d.set_obstacle_region(self.obstacle_region, Q_est)
        self.rrt3d.set_nodes(start, goal)
        tree, parent = self.rrt3d.run()
        self.rrt3d.get_path()
        next_view = self.rrt3d.get_new_point()
        return next_view
    
    
    def my_visualize(self, f_pos, f_axis):
        """
        Visualize the voxel grid, the target and the camera bounds in rviz
        """
        self.rviz_visualizer.visualize_octomap_frame()
        self.rviz_visualizer.visualize_fruit_pose_GT()
        self.rviz_visualizer.visualize_fruit_pose_EST(f_pos, f_axis)
        
        ROI_voxel_points = self.voxel_grid.get_ROI_points()                        # get ROI voxels
        self.rviz_visualizer.visualize_ROI(ROI_voxel_points.cpu().numpy())
        voxel_points, sem_class_ids = self.voxel_grid.get_occupied_points()       # get occupied voxels
        self.rviz_visualizer.visualize_voxels(voxel_points.cpu().numpy(), sem_class_ids.cpu().numpy())
        rospy.loginfo(f"Visualizing No. of voxels: ROI {ROI_voxel_points.shape[0]},  Occupied {voxel_points.shape[0]}")
        if self.verbose_planner:
            print("voxel_points are:",voxel_points)
            print("ROI_voxel_points are:",voxel_points)
    

if __name__ == '__main__':
    viewpoint = np.array([0.5, -0.05, 1.18, 0.70711, 0 ,0, -0.70711])
    print("viewpoint is :", viewpoint)
    position = torch.tensor(viewpoint[:3], dtype=torch.float32)
    orientation = torch.tensor(viewpoint[3:], dtype=torch.float32)
    cam_transform_in_world = transform_from_rotation_translation(orientation[None, :], position[None, :])
    print("cam_transform_in_world is :\n", cam_transform_in_world)
    