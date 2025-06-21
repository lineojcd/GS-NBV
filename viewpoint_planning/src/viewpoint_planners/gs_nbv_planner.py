import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import rospy
from math import exp
import time
from scipy.spatial.transform import Rotation as R
from scene_representation.my_voxel_grid import MyVoxelGrid
from viewpoint_planners.gs_viewpoint_sampler import GSViewpointSampler
from scene_representation.conversions import T_from_rot_trans_np, T_from_rot_trans
from utils.my_rviz_visualizer import MyRvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array, look_at_rotation, my_pause_program
from utils.torch_utils import  transform_from_rotation_translation, transform_from_rot_trans, generate_pixel_coords, get_centroid, transform_points
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation

class GSNBVPlanner(nn.Module):
    """
    Class to plan a locally optimal viewpoint using gradient-based optimization
    """
    def __init__(self, image_size: np.array, intrinsics: np.array, arm_control) -> None:
        """
        :param image_size: size of the image in pixels width/height
        :param intrinsics: camera intrinsic
        """
        super(GSNBVPlanner, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.width=image_size[0]
        self.height=image_size[1]
        self.pick_offset = rospy.get_param("pick_offset", "0.2")           # In meters
        self.use_max_fusion = rospy.get_param("use_max_fusion")
        self.verbose_planner = rospy.get_param("verbose_planner")
        self.R_o_cam = np.array([float(x) for x in rospy.get_param('rot_optical_to_cam').split()])
        self.r_o_pc = np.array([float(x) for x in rospy.get_param('rot_optical_to_pixelcam').split()])
        self.vp_Gsem_filter = rospy.get_param("viewpoint_semantics_gain_filter")                   
        self.vp_motion_cost_filter = rospy.get_param("viewpoint_motion_cost_filter") 
        self.gs_viewpoint_sampler = GSViewpointSampler()
        self.rviz_visualizer = MyRvizVisualizer()
        self.voxel_grid = MyVoxelGrid(self.width, self.height, intrinsics, self.device)
        self.arm_control = arm_control
        self.metric_score = 0
        self.metric_score_history = []
        self.vp_queue = []                  # existing vp in the queue
        self.vp_nbv_history = []            # moved nbv in history
        self.filtered_vp_score_history = []        # checked vp filtered out by IG or close to queue, nbv, previous checked vp
        # self.no_plan_vp_history= []       # No need for now
        self.num_raycasting = 0
        self.all_sampled_vp_history = []
        
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
    
    def viewpoints_sampling(self, f_pos,f_axis, obstacle_points_list, optical_pose):
        # selected_vp_list, valid_vp_list = self.my_viewpoint_sampler.vp_sampler(f_pos,f_axis, obstacle_points_list, optical_pose)
        # selected_vp_list, valid_vp_list = self.my_viewpoint_sampler.uniform_adaptive_sampling(f_pos,f_axis, obstacle_points_list, optical_pose)
        self.gs_viewpoint_sampler.set_geometric_info(f_pos,f_axis, obstacle_points_list, optical_pose)
        selected_vp_list, valid_vp_list = self.gs_viewpoint_sampler.uniform_adaptive_sampling()
        if self.verbose_planner:
            print("vp_candidates_list is \n", selected_vp_list)
        return selected_vp_list, valid_vp_list
    
    def get_num_raycasting(self):
        return self.num_raycasting
    
    def viewpoints_evaluation(self, vp_candidates_list, optical_pose):
        # start = time.time()
        vp_score_list = []
        for vp in vp_candidates_list:       # vp is an np array
            # Calculate Utility by Paper: Efficient Search and Detection of Relevant Plant Parts using Semantics-Aware Active Vision
            
            
            posi = torch.tensor(vp[:3], dtype=torch.float32, device=self.device)
            orien = torch.tensor(R.from_quat(vp[3:]).as_matrix(), dtype=torch.float32, device=self.device)
            vp_Trans_w_o = transform_from_rot_trans(orien[None, :], posi[None, :]) # get viewpose in 4*4 matrice
            vp_IG, G_sem, motion_cost = self.voxel_grid.viewpoint_evaluation(vp_Trans_w_o, optical_pose)
            
            self.num_raycasting += 1
            
            # if not self.viewpoint_filtering(G_sem, motion_cost, posi):
            if not self.viewpoint_filtering(G_sem, motion_cost, vp[:3]):
                vp_score_list.append((vp_IG, G_sem, motion_cost, vp))
            else:
                self.filtered_vp_score_history.append((vp_IG, G_sem, motion_cost, vp))
            
        # print("1 raycasting: ", time.time()-start)
        # my_pause_program()
                
        rospy.loginfo(f"Filtering current sampled viewpoints by Gsem/motion cost ... ")
        return vp_score_list
    
    def viewpoint_filtering(self, G_sem, motion_cost, posi):
        remove_viewpoint = False
        if self.vp_IGfilter_current_iteration(G_sem, motion_cost):
            return True
        if self.vp_filter_in_queue(posi):
            return True
        if self.vp_filter_in_nbv_history(posi):
            return True
        if self.vp_filter_in_filtered_vp_history(posi):
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
    
    def viewpoint_queue_update(self,optical_pose, new_vp_IG_list):
        if len(self.vp_queue) > 0:
            rospy.loginfo(f"Updating motion cost of previous viewpoints")
            self.update_IG_in_queue(optical_pose)
        rospy.loginfo(f"Merging current viewpoints to existing Queue")
        self.vp_queue.extend(new_vp_IG_list)
    
    def viewpoint_queue_sorting(self):
        self.vp_queue.sort(key=lambda x: x[0], reverse=True)
        for vp in self.vp_queue:
            print(f"Score: {vp[0]}, Pose: {vp[-1]}")
    
    def viewpoint_selection(self):
        vp_score = self.vp_queue.pop(0)
        # print("show me nbv: ",nbv)
        # print("show me nbv type: ", type(nbv))
        # nbv_vp = vp_score[-1]
        nbv_vp_score = vp_score
        return nbv_vp_score
    
    def convert_point_to_pose(self, vp_list, f_pos):
        sampled_vp_pose_list = []
        for posi in vp_list:
            orie = my_look_at_rotation(posi, f_pos)
            if self.verbose_planner:
                print(" my_nbv_sampler my_look_at_rotation: ",orie)
            pose = np.concatenate((posi, orie))
            sampled_vp_pose_list.append(pose)
        return sampled_vp_pose_list
    
    def show_all_sampled_viewpoint(self):
        # TODO:
        self.rviz_visualizer.visualize_viewpointslist([ele[-1] for ele in self.all_sampled_vp_history], "all_vps")
        pass
    
    
    def my_next_best_view(self, f_pos, f_axis, obstacle_points_list, optical_pose=None) -> np.array:
        """
        Compute the next best viewpoint
        :return: the NBV viewpoint of camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        rospy.loginfo(f"Sampling valid viewpoint candidates based on sampling radius and estimated fruit position, axis ...")
        

        # start = time.time()
        
        selected_vp_list, valid_vp_list = self.viewpoints_sampling(f_pos,f_axis, obstacle_points_list, optical_pose)

        
        self.rviz_visualizer.visualize_viewpointslist(valid_vp_list,f_pos, "valid_vps")
        # print("show me valid_vp_list", valid_vp_list) # np array: list of [x,y,z]
        self.rviz_visualizer.visualize_viewpointslist([ele[-1] for ele in self.vp_queue],f_pos, "previous_vps")
        
        sampled_vp_pose_list = self.convert_point_to_pose(selected_vp_list, f_pos)
        self.all_sampled_vp_history.extend(selected_vp_list)
        
        rospy.loginfo(f"Evaluating viewpoint candidates ... ")
        
        
        filtered_vp_score_list = self.viewpoints_evaluation(sampled_vp_pose_list, optical_pose)


        self.rviz_visualizer.visualize_viewpointslist([ele[-1] for ele in filtered_vp_score_list],f_pos, "new_vps")
        
        # start = time.time()
        
        reachable_vp_score_list = self.motion_plan_exist_filter(filtered_vp_score_list) # Check motion plan existing filter
        # self.rviz_visualizer.visualize_viewpointslist([ele[-1] for ele in reachable_vp_score_list],f_pos, "reachable_vps")
        # self.no_plan_vp_history.append(vp)
        
        # print("1 raycasting: ", time.time()-start)
        # my_pause_program()
        
        self.viewpoint_queue_update(optical_pose, reachable_vp_score_list)
        
        
        while len(self.vp_queue) == 0:
            rospy.loginfo(f"vp_queue is epmty, re-sample viewpoints points ... ")
            for ele in selected_vp_list:
                try:
                    valid_vp_list.index(ele)
                    valid_vp_list.remove(ele)
                    print("Element is in the list")
                except ValueError:
                    print("Element is not in the list")
                # if ele in valid_vp_list:
                #     valid_vp_list.remove(ele)       # if ele in valid_vp_list:
            new_selected_vp_list = self.gs_viewpoint_sampler.uniform_sampling_on_validvps(valid_vp_list)
            sampled_vp_pose_list = self.convert_point_to_pose(new_selected_vp_list, f_pos)
            rospy.loginfo(f"Re-evaluating viewpoint candidates ... ")
            filtered_vp_score_list = self.viewpoints_evaluation(sampled_vp_pose_list, optical_pose)
            self.rviz_visualizer.visualize_viewpointslist([ele[-1] for ele in filtered_vp_score_list],f_pos, "new_vps")
            reachable_vp_score_list = self.motion_plan_exist_filter(filtered_vp_score_list)
            self.rviz_visualizer.visualize_viewpointslist([ele[-1] for ele in reachable_vp_score_list],f_pos, "reachable_vps")
            self.viewpoint_queue_update(optical_pose, reachable_vp_score_list)
        
        
        # start = time.time()
        
        self.viewpoint_queue_sorting()
        nbv_vp_score = self.viewpoint_selection()  
        self.vp_nbv_history.append(nbv_vp_score)
        nbv_vp = nbv_vp_score[-1]
        
        # print("1 raycasting: ", time.time()-start)
        # my_pause_program()
        
        
        
        
        self.rviz_visualizer.visualize_viewpointslist([ele[-1] for ele in self.filtered_vp_score_history],f_pos, "filtered_vps")
        self.rviz_visualizer.visualize_viewpointslist([nbv_vp],f_pos, "nbv_vp")
        return nbv_vp

    def my_visualize(self, f_pos, f_axis):
        """
        Visualize the voxel grid, the target and the camera bounds in rviz
        """
        self.rviz_visualizer.visualize_octomap_frame()
        self.rviz_visualizer.visualize_fruit_pose_GT()
        self.rviz_visualizer.visualize_fruit_pose_EST(f_pos, f_axis)
        self.rviz_visualizer.visualize_picking_ring_GT()
        
        ROI_region_voxel_points = self.voxel_grid.get_ROI_points()                        # get ROI voxels
        self.rviz_visualizer.visualize_ROI(ROI_region_voxel_points.cpu().numpy())
        
        # non_ROI_voxel_points, _ = self.voxel_grid.get_non_ROI_occupied_points()       # get non ROI occupied voxels
        # self.rviz_visualizer.visualize_voxels(non_ROI_voxel_points.cpu().numpy(), None)
        
        
        
        voxel_points, sem_class_ids = self.voxel_grid.get_occupied_points()       # get occupied voxels
        self.rviz_visualizer.visualize_voxels(voxel_points.cpu().numpy(), sem_class_ids.cpu().numpy())
        
        rospy.loginfo(f"Visualizing No. of voxels: ROI {ROI_region_voxel_points.shape[0]},  Occupied {voxel_points.shape[0]}")
        if self.verbose_planner:
            print("voxel_points are:",voxel_points)
            print("ROI_region_voxel_points are:",ROI_region_voxel_points)
    

if __name__ == '__main__':
    viewpoint = np.array([0.5, -0.05, 1.18, 0.70711, 0 ,0, -0.70711])
    print("viewpoint is :", viewpoint)
    position = torch.tensor(viewpoint[:3], dtype=torch.float32)
    orientation = torch.tensor(viewpoint[3:], dtype=torch.float32)
    cam_transform_in_world = transform_from_rotation_translation(orientation[None, :], position[None, :])
    print("cam_transform_in_world is :\n", cam_transform_in_world)