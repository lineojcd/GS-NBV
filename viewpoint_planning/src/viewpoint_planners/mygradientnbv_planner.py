import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import rospy
from scene_representation.gradient_voxel_grid import GradientVoxelGrid
import time

from utils.gradient_rviz_visualizer import GradientRvizVisualizer
# from utils.my_rviz_visualizer import MyRvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation, transform_from_rot_trans
from scipy.spatial.transform import Rotation as R
from scene_representation.my_voxel_grid import MyVoxelGrid
from viewpoint_planners.gs_viewpoint_sampler import GSViewpointSampler
from scene_representation.conversions import T_from_rot_trans_np, T_from_rot_trans
from utils.py_utils import numpy_to_pose, numpy_to_pose_array, look_at_rotation, my_pause_program
from utils.torch_utils import  transform_from_rotation_translation, transform_from_rot_trans, generate_pixel_coords, get_centroid, transform_points
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation

class GradientNBVPlanner(nn.Module):
    """
    Class to plan a locally optimal viewpoint using gradient-based optimization
    """
    def __init__(self,image_size: np.array,intrinsics: np.array,start_pose: np.array,
        arm_control, target_params: np.array = np.array([0.5, -0.4, 1.1]),) -> None:
        """
        Initialize the planner
        :param image_size: size of the image in pixels
        :param intrinsics: camera intrinsic
        """
        super(GradientNBVPlanner, self).__init__()
        self.start_pose = start_pose
        self.num_samples = rospy.get_param("GNBV_num_samples", "1")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.width=image_size[0]
        self.height=image_size[1]
        self.lr = rospy.get_param("GNBV_learning_rate", "0.03")           # In meters
        self.use_max_fusion = rospy.get_param("use_max_fusion")
        self.verbose_planner = rospy.get_param("verbose_planner")
        self.R_o_cam = np.array([float(x) for x in rospy.get_param('rot_optical_to_cam').split()])
        self.r_o_pc = np.array([float(x) for x in rospy.get_param('rot_optical_to_pixelcam').split()])
        self.fruit_posi_GT = [float(x) for x in rospy.get_param('fruit_posi_GT').split()] 
        self.voxel_size = rospy.get_param("voxel_size", "0.003")
        self.roi_range = [float(x) * self.voxel_size for x in rospy.get_param('roi_range').split()]
        self.vp_sampling_Rmin = rospy.get_param("SCNBV_sampling_Rmin", "0.21")
        self.arm_control = arm_control
        self.metric_score = 0
        self.metric_score_history = []
        self.vp_queue = []                  # existing vp in the queue
        self.vp_nbv_history = []            # moved nbv in history
        self.filtered_vp_score_history = []        # checked vp filtered out by IG or close to queue, nbv, previous checked vp
        # self.no_plan_vp_history= []       # No need for now
        self.num_raycasting = 0
        # self.rviz_visualizer = SCRvizVisualizer()
        self.rviz_visualizer = GradientRvizVisualizer()
        self.all_sampled_vp_history = []
        
        # Initialize the optimization parameters
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # self.optimization_params()
        self.voxel_grid = GradientVoxelGrid(self.width, self.height, intrinsics, self.device)
        
        
    # def optimization_params(self, start_pose: np.array, target_params: np.array) -> None:
    def optimization_params(self) -> None:
        """
        Initialize the optimization parameters
        """
        # self.camera_params = nn.Parameter(
        #     torch.tensor([start_pose[0],start_pose[1],start_pose[2],target_params[0],target_params[1],target_params[2]],
        #                 dtype=torch.float32, device=self.device, requires_grad=True))
        # self.target_params = torch.tensor(target_params,dtype=torch.float32,device=self.device)
        # self.camera_bounds = torch.tensor(
        #     [[self.start_pose[0] - 0.2,
        #     self.start_pose[1] - 0.1,
        #     self.start_pose[2] - 0.15,
        #     self.fruit_posi_GT[0] - self.roi_range[0],self.fruit_posi_GT[1] - self.roi_range[1],self.fruit_posi_GT[2] - self.roi_range[2],
        #     ],[
        #     self.start_pose[0] + 0.2,
        #     self.start_pose[1] + 0.1,
        #     self.start_pose[2] + 0.15,
        #     self.fruit_posi_GT[0] + self.roi_range[0],self.fruit_posi_GT[1] + self.roi_range[1],self.fruit_posi_GT[2] + self.roi_range[2],
        #     ]],dtype=torch.float32,device=self.device,)
        
        # self.camera_bounds = torch.tensor(
        #     [[self.fruit_posi_GT[0] - self.vp_sampling_Rmin,
        #     self.fruit_posi_GT[1]  - 0.148, # 0.148 = 0.21/sqrt(2)
        #     self.fruit_posi_GT[2] - self.vp_sampling_Rmin,
        #     self.fruit_posi_GT[0] - self.roi_range[0],self.fruit_posi_GT[1] - self.roi_range[1],self.fruit_posi_GT[2] - self.roi_range[2],
        #     ],[
        #     self.fruit_posi_GT[0] + self.vp_sampling_Rmin,
        #     self.fruit_posi_GT[1] + self.vp_sampling_Rmin + 0.1,
        #     self.fruit_posi_GT[2] + self.vp_sampling_Rmin,
        #     self.fruit_posi_GT[0] + self.roi_range[0],self.fruit_posi_GT[1] + self.roi_range[1],self.fruit_posi_GT[2] + self.roi_range[2],
        #     ]],dtype=torch.float32,device=self.device,)
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.03)
        pass

    def set_camera_params(self,f_pos):
        self.camera_params = nn.Parameter(torch.tensor([self.start_pose[0],self.start_pose[1],self.start_pose[2],
            f_pos[0],f_pos[1],f_pos[2]],dtype=torch.float32, device=self.device, requires_grad=True))
        
        self.camera_bounds = torch.tensor(
            [[f_pos[0] - self.vp_sampling_Rmin - 0.05,
            f_pos[1]  - 0.148, # 0.148 = 0.21/sqrt(2)
            f_pos[2] - self.vp_sampling_Rmin + 0.1,
            f_pos[0] - self.roi_range[0],f_pos[1] - self.roi_range[1],f_pos[2] - self.roi_range[2],
            ],[
            # f_pos[0] + self.vp_sampling_Rmin + 0.05,
            f_pos[0] + self.vp_sampling_Rmin + 0.05,
            # f_pos[1] + self.vp_sampling_Rmin + 0.2,
            self.start_pose[1],
            f_pos[2] + self.vp_sampling_Rmin,
            f_pos[0] + self.roi_range[0],f_pos[1] + self.roi_range[1],f_pos[2] + self.roi_range[2],
            ]],dtype=torch.float32,device=self.device,)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # Visualize camera bounds
        camera_bounds = self.camera_bounds.cpu().numpy()[:, :3]
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds)
        
        # print("camera_bounds is : ", camera_bounds)
        # print("f_pos is : ", f_pos)
        # print("self.start_pose[1] is : ", self.start_pose[1])
        # my_pause_program()
        
        

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
    
    def check_trajectory_exist(self, pose):
        return self.arm_control.check_trajectory_exist(pose)
    
    def loss(self, target_pos: np.array) -> torch.tensor:
        """
        Compute the loss for the current viewpoint
        :return: loss
        """
        if target_pos is not None:
            self.target_params = torch.tensor(target_pos, dtype=torch.float32, device=self.device)
        else:
            self.target_params = self.camera_params[3:]
        # start=time.time()
        loss, gain_image = self.voxel_grid.compute_gain(self.camera_params[:3], self.target_params)
        # print(">> raycast cost: ", time.time() - start)
        # my_pause_program()
        return loss, gain_image

    def next_best_view(self, target_pos=None) -> Tuple[np.array, float, int]:
        """
        Compute the next best viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        :return: loss
        :return: number of samples
        """
        

        
        for _ in range(self.num_samples):
            self.optimizer.zero_grad()
            # loss, gain_image = self.loss(target_pos)
            loss, gain_image = self.loss(None)
            
            
            
            loss.backward()
            self.optimizer.step()
            self.camera_params.data = torch.clamp(self.camera_params.data, self.camera_bounds[0], self.camera_bounds[1])
            self.num_raycasting+=1
        
        nbv = self.get_viewpoint()

        
        # start=time.time()
        if self.check_trajectory_exist(nbv):
            self.rviz_visualizer.visualize_viewpointslist([nbv],target_pos, "nbv_vp")
        else:
            print("pose is not reachable")
            nbv = None
        
        # print("&&&raycast cost: ", time.time() - start)
        # my_pause_program()
        

        
        # self.rviz_visualizer.visualize_viewpoint(numpy_to_pose(nbv))
        # self.rviz_visualizer.visualize_viewpointslist([nbv],self.start_pose[:3], "nbv_vp")
        self.rviz_visualizer.visualize_gain_image(gain_image)
        loss = loss.detach().cpu().numpy()
        return nbv, loss, self.num_samples

    def get_viewpoint(self) -> np.array:
        """
        Get the current viewpoint
        :return: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        # quat = look_at_rotation(self.camera_params[:3], self.camera_params[3:])
        cam_posi = self.camera_params.detach().cpu().numpy()[:3]
        tar_posi = self.camera_params[3:].detach().cpu().numpy()
        quat = my_look_at_rotation(cam_posi, tar_posi)
        # orie = my_look_at_rotation(camera_posi.detach().cpu().numpy(), target_params.detach().cpu().numpy())
        # quat = quat.detach().cpu().numpy()
        viewpoint = np.zeros(7)
        # viewpoint[:3] = self.camera_params.detach().cpu().numpy()[:3]
        viewpoint[:3] = cam_posi
        viewpoint[3:] = quat
        return viewpoint
    
    def get_num_raycasting(self):
        return self.num_raycasting

    def get_occupied_points(self):
        voxel_points, sem_conf_scores, sem_class_ids = (self.voxel_grid.get_occupied_points())
        voxel_points = voxel_points.cpu().numpy()
        sem_conf_scores = sem_conf_scores.cpu().numpy()
        sem_class_ids = sem_class_ids.cpu().numpy()
        return voxel_points, sem_conf_scores, sem_class_ids

    def visualize(self):
        """
        Visualize the voxel grid, the target and the camera bounds in rviz
        """
        voxel_points, sem_conf_scores, sem_class_ids = self.get_occupied_points()
        self.rviz_visualizer.visualize_voxels(voxel_points, sem_conf_scores, sem_class_ids)
        
        # Visualize target
        # target = self.target_params.detach().cpu().numpy()
        target = self.camera_params.detach().cpu().numpy()[3:]
        rois = np.array([[*target, 1.0, 0.0, 0.0, 0.0]])
        self.rviz_visualizer.visualize_rois(numpy_to_pose_array(rois))
        # Visualize camera bounds
        camera_bounds = self.camera_bounds.cpu().numpy()[:, :3]
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds)
    


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
        # # Visualize camera bounds
        # camera_bounds = self.camera_bounds.cpu().numpy()[:, :3]
        # self.rviz_visualizer.visualize_camera_bounds(camera_bounds)
        if self.verbose_planner:
            print("voxel_points are:",voxel_points)
            print("ROI_voxel_points are:",voxel_points)

