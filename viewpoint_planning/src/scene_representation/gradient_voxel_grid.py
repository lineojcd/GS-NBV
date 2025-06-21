"""
Author: Akshay K. Burusa
Maintainer: Akshay K. Burusa
"""
import rospy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rospy
from scene_representation.raysampler import RaySampler
from scene_representation.my_raysampler import MyRaySampler
from scene_representation.my_raycast_algo_3d import raycast_3d, check_visibility,check_semantics_visibility_tensor
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation,get_frustum_points, transform_from_rot_trans
from scene_representation.conversions import T_from_rot_trans_np
from scipy.spatial.transform import Rotation as R
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation

class GradientVoxelGrid:
    """
    3D representation to store occupancy info & other features (e.g. semantics) over multiple viewpoints
    """
    def __init__(self,width: int,height: int,intrinsics: torch.tensor,device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        """
        Constructor
        :param width: image width
        :param height: image height
        :param device: device to use for computation
        """
        self.device = device
        self.width = width
        self.height = height
        # self.target_params = target_params
        self.min_depth = rospy.get_param("near_clipping_plane", "0.03")
        self.max_depth = rospy.get_param("far_clipping_plane", "0.72")
        self.verbose_octomap = rospy.get_param("verbose_octomap")
        self.fruit_posi_GT = [float(x) for x in rospy.get_param('fruit_posi_GT').split()]
        self.fruit_posi_GT = torch.tensor(self.fruit_posi_GT, dtype=torch.float32,device=self.device)
        self.grid_range =torch.tensor([float(x) for x in rospy.get_param('grid_range').split()], dtype=torch.float32, device=self.device)
        self.intrinsics =  torch.tensor(intrinsics,dtype=torch.float32,device=self.device)
        self.voxel_size = torch.tensor(rospy.get_param("voxel_size", "0.003"), dtype=torch.float32, device=self.device)
        self.voxel_dims = (self.grid_range / self.voxel_size).long()       # index range of each voxel
        rospy.loginfo(f"Effective Octomap range: {self.grid_range}; Total No, of voxels: {self.voxel_dims[0]*self.voxel_dims[1]*self.voxel_dims[2]}")
        self.r_o_pc = np.array([float(x) for x in rospy.get_param('rot_optical_to_pixelcam').split()])
        self.T_o_pc = T_from_rot_trans_np(R.from_quat(self.r_o_pc).as_matrix(), np.zeros((1, 3)))
        self.T_o_pc = torch.as_tensor(self.T_o_pc, dtype=torch.float32, device=self.device)      
        self.num_pts_per_ray = rospy.get_param("num_pts_per_ray", "128")
        self.num_features_per_voxel = rospy.get_param("num_features_per_voxel", "4")
        self.eps = torch.tensor(1e-7, dtype=torch.float32,device=self.device)          # epsilon value for numerical stability
        self.evaluation_metric = rospy.get_param("evaluation_metric")                   #semantic_class/coverage
        self.lamda = rospy.get_param("move_cost_coefficient", "-1")                   
        self.occupied_lodds = float(rospy.get_param("occupied_lodds", "2.2"))
        self.free_lodds = float(rospy.get_param("free_lodds", "-1.4"))
        self.init_ray_occ_lodds = float(rospy.get_param("init_ray_occ_lodds", "-1.4"))
        self.init_ray_sem_conf_lodds = float(rospy.get_param("init_ray_sem_conf_lodds", "0.0"))   
        self.init_sem_cls_id = rospy.get_param("init_sem_cls_id", "-1")             
        # self.voxel_max_fusion_coef = rospy.get_param("voxel_max_fusion_coefficient", "0.9")             
        self.voxel_max_fusion_coef = rospy.get_param("GNBV_voxel_max_fusion_coefficient", "1.00")             
        self.roi_range = [int(x) for x in rospy.get_param('roi_range').split()]
        self.voxel_occupancy_threshold = rospy.get_param("voxel_occupancy_threshold", "0.50") 
        self.origin = torch.tensor([float(x) for x in rospy.get_param('voxel_origin').split()], dtype=torch.float32, device=self.device)
        self.voxel_grid = torch.zeros((self.voxel_dims[0], self.voxel_dims[1], self.voxel_dims[2],
                                    self.num_features_per_voxel), dtype=torch.float32, device=self.device) # 4D voxel grid: W*H*D*4
        self.voxel_roi_layer = rospy.get_param("voxel_roi_layer", "0")  
        self.voxel_occ_prob_layer = rospy.get_param("voxel_occ_prob_layer", "1")  
        self.voxel_sem_conf_layer = rospy.get_param("voxel_sem_conf_layer", "2")  
        self.voxel_sem_cls_layer = rospy.get_param("voxel_sem_cls_layer", "3")  
        self.roi_sem_conf = rospy.get_param("roi_sem_conf", "0.50") 
        self.roi_occ_prob = rospy.get_param("roi_occ_prob", "0.50") 
        self.voxel_grid[..., self.voxel_occ_prob_layer] = rospy.get_param("init_occ_prob", "0.5")   # Init occ prob as 0.5: most uncertain at the start 
        self.voxel_grid[..., self.voxel_sem_conf_layer] = self.eps                                  # Init sem conf prob close to 0
        self.voxel_grid[..., self.voxel_sem_cls_layer] = self.init_sem_cls_id                       # Init sem cls to bg: -1 
        self.target_bounds = None   # Define ROI around the target
        self.set_my_target_roi_region(self.fruit_posi_GT) # Define ROI around the fuir; ROI might be: 0 avo; 1 peduncle; 2 cont; 3 env
        
        # Occ prob along the ray is init to 0.2, which gives a log odds of -1.4
        ray_occ = self.init_ray_occ_lodds * torch.ones((self.num_pts_per_ray, 1),dtype=torch.float32,device=self.device) # dimensions[128, 1]
        self.ray_occ = ray_occ.unsqueeze(0).repeat(width * height, 1, 1 )       # (W x H, num_pts_per_ray, 1)
        # Seman conf prob along the ray is init to 0.2, which gives log odds of -1.4
        ray_sem_conf = self.init_ray_sem_conf_lodds * torch.ones(self.num_pts_per_ray,dtype=torch.float32,device=self.device)
        # Seman cls id along the ray is init to -1 (background)
        ray_sem_cls = self.init_sem_cls_id * torch.ones(self.num_pts_per_ray,dtype=torch.float32,device=self.device)
        ray_sem = torch.stack((ray_sem_conf, ray_sem_cls), dim=-1)
        self.ray_sem = ray_sem.unsqueeze(0).repeat(width * height, 1, 1 )  # (W x H, num_pts_per_ray, 2)
        self.t_vals = torch.linspace(0.0,1.0,self.num_pts_per_ray,dtype=torch.float32,device=self.device)
        self.explored_semantic_voxels = 0 
        self.explored_semantic_voxels_update = 0
        
        self.gnbv_origin = self.fruit_posi_GT - self.grid_range / 2.0
        self.min_bound = self.origin
        # self.min_bound = self.gnbv_origin
        self.max_bound = self.origin + self.grid_range
        # self.max_bound = self.gnbv_origin + self.grid_range
        # Define regions of interest around the target
        # Use self.set_my_target_roi_region 
        # self.set_target_roi(target_params)
        
        # Ray sampler
        # self.ray_sampler = RaySampler(width=width,height=height,device=device)
        

    def insert_depth_and_semantics(self,depth_image: torch.tensor,semantics: torch.tensor,transforms: torch.tensor,) -> None:
        """
        Insert a point cloud into the voxel grid
        :param depth_image: depth image from the current viewpoint (W x H)
        :param semantics: semantic confidences and labels from the current viewpoint (2 x W x H)
        :param position: position of the current viewpoint (3,)
        :param orientation: orientation of the current viewpoint (4,)
        :return: None
        """
        # Convert depth image to point cloud
        (ray_origins,ray_directions,points_mask,) = self.ray_sampler.ray_origins_directions(depth_image=depth_image, transforms=transforms)
        ray_points = (ray_directions[:, :, None, :] * self.t_vals[None, :, None] + ray_origins[:, :, None, :]).view(-1, 3)

        # Convert point cloud to voxel grid coordinates
        grid_coords = torch.div(ray_points - self.origin, self.voxel_size, rounding_mode="floor")
        valid_indices = self.get_valid_indices_from_octomap(grid_coords, self.voxel_dims)
        gx, gy, gz = grid_coords[valid_indices].to(torch.long).unbind(-1)
        # Get the log odds of the occupancy and semantic probabilities
        log_odds = torch.log(torch.div(self.voxel_grid[gx, gy, gz, 1:3], 1.0 - self.voxel_grid[gx, gy, gz, 1:3]))
        # Update the log odds of the occupancy probabilities
        ray_occ = self.ray_occ.clone()
        ray_occ[:, -2:, :] = points_mask.permute(1, 0).repeat(1, 2).unsqueeze(-1)
        log_odds[..., 0] += ray_occ.view(-1, 1)[valid_indices, -1]
        # Update the log odds of the semantic probabilities
        ray_sem = self.ray_sem.clone()
        ray_sem[..., -1, :] = semantics.view(-1, 2)
        ray_sem = ray_sem.view(-1, 2)
        log_odds[..., 1] += ray_sem[valid_indices, 0]
        # Convert the log odds back to occupancy and semantic probabilities
        odds = torch.exp(log_odds)
        self.voxel_grid[gx, gy, gz, 1:3] = torch.div(odds, 1.0 + odds)
        self.voxel_grid[..., 1:3] = torch.clamp(self.voxel_grid[..., 1:3], self.eps, 1.0 - self.eps)
        
        # assign class id to voxel_grid
        self.voxel_grid[gx, gy, gz, 3] = ray_sem[valid_indices, 1]
        # Check the values within the target bounds and count the number of updated voxels
        if self.target_bounds is not None:
            target_voxels = self.voxel_grid[
                self.target_bounds[0] : self.target_bounds[3],
                self.target_bounds[1] : self.target_bounds[4],
                self.target_bounds[2] : self.target_bounds[5], 2,]
            coverage = torch.sum((target_voxels != 0.5)) / target_voxels.numel() * 100
            return coverage

    def compute_gain(self,camera_posi: torch.tensor,target_params: torch.tensor) -> torch.tensor:
        """
        Compute the gain for a given set of parameters
        :param camera_posi: camera position
        :param target_params: target parameters
        :param current_params: current parameters
        :return: total gain for the viewpoint defined by the parameters
        """
        # quat = look_at_rotation(camera_posi, target_params)
        # transforms = transform_from_rotation_translation(quat[None, :], camera_posi[None, :])
        # ray_origins, ray_directions, _ = self.ray_sampler.ray_origins_directions(transforms=transforms)
        
        # Compute point cloud by ray-tracing along ray origins and directions
        t_vals = self.t_vals.clone().requires_grad_()
        
        # ray_points = (
        #     ray_directions[:, :, None, :] * t_vals[None, :, None]
        #     + ray_origins[:, :, None, :]
        # ).view(-1, 3)
        
        # posi = torch.tensor(vp[:3], dtype=torch.float32, device=self.device)
        # print(camera_posi)
        # print(target_params)
        orie = my_look_at_rotation(camera_posi.detach().cpu().numpy(), target_params.detach().cpu().numpy())
        # posi = torch.tensor(camera_posi, dtype=torch.float32, device=self.device)
        # orien = torch.tensor(R.from_quat(vp[3:]).as_matrix(), dtype=torch.float32, device=self.device)
        orien = torch.tensor(R.from_quat(orie).as_matrix(), dtype=torch.float32, device=self.device)
        vp_Trans_w_o = transform_from_rot_trans(orien[None, :], camera_posi[None, :]) # get viewpose in 4*4 matrice
        
        rospy.loginfo(f"Geting point clouds from FOV")
        # Convert depth image to point cloud (in Wf), and get its occ log-odds map
        (ray_origins, ray_directions, _) = get_frustum_points(None,     # not unit direction, it has length
            self.T_o_pc, vp_Trans_w_o, self.min_depth, self.max_depth,self.width, self.height,
            self.intrinsics, self.occupied_lodds, self.free_lodds, self.device)
        # frustum 3D space: contain points from near to far and span 128 depth interval in world_frame
        ray_points = (ray_directions[:, :, None, :] * t_vals[None, :, None] + ray_origins[:, :, None, :]).view(-1, 3)

        ray_points_nor = self.normalize_3d_coordinate(ray_points)
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3).repeat(2, 1, 1, 1, 1)
        # Sample the occupancy probabilities and semantic confidences along each ray
        grid = self.voxel_grid[None, ..., 1:3].permute(4, 0, 1, 2, 3)
        occ_sem_confs = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem_confs = occ_sem_confs.view(2, -1, self.num_pts_per_ray)
        occ_sem_confs = occ_sem_confs.clamp(self.eps, 1.0 - self.eps)
        # Compute the entropy of the semantic confidences along each ray
        opacities = torch.sigmoid(1e7 * (occ_sem_confs[0, ...] - 0.51))
        transmittance = self.shifted_cumprod(1.0 - opacities)
        ray_gains = transmittance * self.entropy(occ_sem_confs[1, ...])
        # Create a gain image for visualization
        gain_image = ray_gains.view(-1, self.num_pts_per_ray).sum(1)
        gain_image = gain_image.view(self.height, self.width)
        gain_image = gain_image - gain_image.min()
        # gain_image = gain_image / gain_image.max()
        gain_image = gain_image / 32.0
        gain_image = gain_image.detach().cpu().numpy()
        gain_image = plt.cm.viridis(gain_image)[..., :3]
        # Compute the semantic gain
        semantic_gain = torch.log(torch.mean(ray_gains) + self.eps)
        loss = -semantic_gain
        return loss, gain_image

    def entropy(self, probs: torch.tensor) -> torch.tensor:
        """
        Compute the entropy of a set of probabilities
        :param probs: tensor of probabilities
        :return: tensor of entropies
        """
        probs_inv = 1.0 - probs
        gains = -(probs * torch.log2(probs)) - (probs_inv * torch.log2(probs_inv))
        return gains

    def set_target_roi(self, target_params: torch.tensor) -> None:
        # Define regions of interest around the target
        if target_params is None:
            return
        target_coords = torch.div(
            target_params - self.origin, self.voxel_size, rounding_mode="floor"
        ).to(torch.long)
        x_min = torch.clamp(target_coords[0] - 25, 0, self.voxel_dims[0])
        x_max = torch.clamp(target_coords[0] + 25, 0, self.voxel_dims[0])
        y_min = torch.clamp(target_coords[1] - 25, 0, self.voxel_dims[1])
        y_max = torch.clamp(target_coords[1] + 25, 0, self.voxel_dims[1])
        z_min = torch.clamp(target_coords[2] - 25, 0, self.voxel_dims[2])
        z_max = torch.clamp(target_coords[2] + 25, 0, self.voxel_dims[2])
        self.voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max, 0] = 1
        self.voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max, 2] = 0.5
        self.target_bounds = torch.tensor([x_min, y_min, z_min, x_max, y_max, z_max], device=self.device)
        self.voxel_grid[..., 1:3] = torch.clamp(self.voxel_grid[..., 1:3], self.eps, 1.0 - self.eps)

    def normalize_3d_coordinate(self, points):
        """
        Normalize a tensor of 3D points to the range [-1, 1] along each axis.
        :param points: tensor of 3D points of shape (N, 3)
        :return: tensor of normalized 3D points of shape (N, 3)
        """
        
        # TODO
        
        # Compute the range of values for each dimension
        x_min, y_min, z_min = self.min_bound
        x_max, y_max, z_max = self.max_bound
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        # Normalize the points to the range [-1, 1]
        n_points = points.clone()
        n_points_out = torch.zeros_like(n_points)
        n_points_out[..., 0] = 2.0 * (n_points[..., 2] - z_min) / z_range - 1.0
        n_points_out[..., 1] = 2.0 * (n_points[..., 1] - y_min) / y_range - 1.0
        n_points_out[..., 2] = 2.0 * (n_points[..., 0] - x_min) / x_range - 1.0
        return n_points_out

    def shifted_cumprod(self, x: torch.tensor, shift: int = 1) -> torch.tensor:
        """
        Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of ones and removes
        `shift` trailing elements to/from the last dimension of the result
        :param x: tensor of shape (N, ..., C)
        :param shift: number of elements to prepend/remove
        :return: tensor of shape (N, ..., C)
        """
        x_cumprod = torch.cumprod(x, dim=-1)
        x_cumprod_shift = torch.cat(
            [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
        )
        return x_cumprod_shift

    def get_occupied_points(self):
        """
        Returns the coordinates of the occupied points in the grid
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """
        grid_coords = torch.nonzero(self.voxel_grid[..., self.voxel_occ_prob_layer] > self.voxel_occupancy_threshold)
        # semantics_conf = self.voxel_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 2]
        class_ids = self.voxel_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 3]
        points = grid_coords * self.voxel_size + self.origin
        # return points, class_ids
        return points,  class_ids
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def get_ROI_points(self):
        """
        Returns the coordinates of the occupied points in the grid
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """
        grid_coords = torch.nonzero(self.voxel_grid[..., self.voxel_roi_layer] == 1 )
        return grid_coords * self.voxel_size + self.origin

    def get_non_ROI_occupied_points(self):
        """
        Returns the coordinates of the occupied points in the grid
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """
        # All points that meet the occupancy threshold (grid_coords)
        grid_mask = self.voxel_grid[..., self.voxel_occ_prob_layer] > self.voxel_occupancy_threshold
        # Create an empty mask of the same shape, initially set to True for non-ROI
        non_roi_mask = grid_mask.clone()
        # Mark ROI region as False (because these are the ROI points)
        non_roi_mask[self.target_bounds[0]:self.target_bounds[3],
                    self.target_bounds[1]:self.target_bounds[4],
                    self.target_bounds[2]:self.target_bounds[5]] = False
        # Get the coordinates of the non-ROI points
        non_roi_points = torch.nonzero(self.voxel_grid[non_roi_mask, self.voxel_occ_prob_layer] > self.voxel_occupancy_threshold)
        points = non_roi_points * self.voxel_size + self.origin
        return points, class_ids
        return points, None

    def get_valid_indices_from_octomap(self, grid_coords: torch.tensor, dims: torch.tensor) -> torch.tensor:
        """
        Get the indices of the grid coordinates that are within the grid bounds
        :param grid_coords: tensor of grid coordinates
        :param dims: tensor of grid dimensions
        :return: tensor of valid indices
        """
        valid_indices = ((grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < dims[0])
                            & (grid_coords[:, 1] >= 0)& (grid_coords[:, 1] < dims[1])
                            & (grid_coords[:, 2] >= 0)& (grid_coords[:, 2] < dims[2]))
        return valid_indices
    
    def update_occupancy_probability(self, current_occ_log_odds,  new_occupancy_score_map, update_indices):
        rospy.loginfo("Updating the log odds of the occupancy probabilities")
        ray_occ = self.ray_occ.clone()                            # self.ray_occ shape: (W x H, num_pts_per_ray, 1)
        
        # repeats the mask for two of the last points along each ray
        # ray_occ[:, -2:, :] = occupancy_score_map.permute(1, 0).repeat(1, 2).unsqueeze(-1)
        rospy.logwarn("In GNBV, it repeats the last pt twice; I changed it to once")
        ray_occ[:, -1:, :] = new_occupancy_score_map.permute(1, 0).unsqueeze(-1)
        current_occ_log_odds += ray_occ.view(-1, 1)[update_indices, -1]
        updated_occ_odds = torch.exp(current_occ_log_odds)
        return updated_occ_odds

    def update_semantic_class_and_probability(self, gx,gy,gz,update_indices,  new_semantics):
        rospy.loginfo("Updating the semantic class id and log odds of the semantic probabilities")
        ray_sem = self.ray_sem.clone()                          # (W x H, num_pts_per_ray, 2); 
        if new_semantics is not None:
            ray_sem[..., -1, :] = new_semantics.view(-1, 2)             # 52428800 * 2
        else:
            rospy.logwarn("There is no new semantics in this view.")
        ray_sem = ray_sem.view(-1, 2)
        
        rospy.loginfo(f"Applying Max fusion algo voxel-wise on the semantics")
        # from paper: Efficient Search and Detection of Relevant PlantParts using Semantics-Aware Active Vision
        # Extract the curr cls ID and semantic conf from the voxel grid
        current_sem_conf = self.voxel_grid[gx, gy, gz, self.voxel_sem_conf_layer]
        current_class_id = self.voxel_grid[gx, gy, gz, self.voxel_sem_cls_layer]
        
        # New class ID and semantic confidence from the current ray
        new_class_id = ray_sem[update_indices, 1]
        new_sem_conf_log_odds = ray_sem[update_indices, 0]                   # conf_score log_odds
        new_sem_conf_odds = torch.exp(new_sem_conf_log_odds)
        new_sem_conf = torch.div(new_sem_conf_odds, 1.0 + new_sem_conf_odds)
        rospy.logwarn("You may consider to directly operation on semantic probability in the future")
        
        if self.verbose_octomap:
            # Get unique values from the tensor
            unique_values_cur = torch.unique(current_class_id)
            unique_values_new = torch.unique(new_class_id)
            print("unique_values_cur: ", unique_values_cur)
            print("unique_values_new: ", unique_values_new)
            
        # Initialize updated class ID and confidence
        updated_class_id = current_class_id.clone()
        updated_conf = current_sem_conf.clone()
        
        # Create a mask for where the class IDs are the same
        same_id_mask = current_class_id == new_class_id
        # Average conf scores where the class IDs are the same
        updated_conf[same_id_mask] = (current_sem_conf[same_id_mask] + new_sem_conf[same_id_mask]) / 2
        
        # Mask for when the new semantic conf is >= the current conf, across different classes
        higher_conf_diff_class_mask = ~same_id_mask & (new_sem_conf >= current_sem_conf)
        # Update class IDs and confidence scores where the new confidence is higher for different classes
        updated_class_id[higher_conf_diff_class_mask] = new_class_id[higher_conf_diff_class_mask]
        updated_conf[higher_conf_diff_class_mask] = self.voxel_max_fusion_coef* new_sem_conf[higher_conf_diff_class_mask]

        # Update the voxel grid with the new class ID and semantic confidence
        self.voxel_grid[gx, gy, gz, self.voxel_sem_cls_layer] = updated_class_id    # Update class ID
        self.voxel_grid[gx, gy, gz, self.voxel_sem_conf_layer] = updated_conf        # Update semantic confidence
        
    def insert_depth_and_semantics_max_fusion(self,depth_img: torch.tensor,semantics: torch.tensor,Trans_w_o: torch.tensor) -> None:
        """
        Insert a point cloud into the voxel grid 
        :param depth_image: depth image from the current viewpoint (W x H)
        :param semantics: semantic confidences and labels from the current viewpoint (2 x W x H): [score, label] 
        :param position: position of the current viewpoint (3,) in world frame
        :param orientation: orientation of the current viewpoint (4,) in world frame
        :return: None
        """
        rospy.loginfo(f"Geting point clouds from FOV")
        # Convert depth image to point cloud (in Wf), and get its occ log-odds map
        (ray_origins,ray_directions,occupancy_score_map) = get_frustum_points(depth_img,     # not unit direction, it has length
            self.T_o_pc, Trans_w_o, self.min_depth, self.max_depth,self.width, self.height,
            self.intrinsics, self.occupied_lodds, self.free_lodds, self.device)
        
        # frustum 3D space: contain points from near to far and span 128 depth interval in world_frame
        ray_points = (ray_directions[:, :, None, :] * self.t_vals[None, :, None] + ray_origins[:, :, None, :]).view(-1, 3)
        
        rospy.loginfo(f"Converting point clouds from world to voxel frame, voxel size is: {round(self.voxel_size.item(),3)} meter")
        # grid_coords: A tensor containing grid coordinates. Each row represents a 3D coords (x, y, z).
        grid_coords = torch.div(ray_points - self.origin, self.voxel_size, rounding_mode="floor")  
        
        if self.verbose_octomap:
            print("ray_points in insert_depth_and_semantics_max_fusion are", ray_points)
            print("grid_coords in insert_depth_and_semantics_max_fusion are", grid_coords)
        
        # A Boolean tensor indicating if each coordinate is within the specified bounds in (index)
        valid_indices = self.get_valid_indices_from_octomap(grid_coords, self.voxel_dims)
        gx, gy, gz = grid_coords[valid_indices].to(torch.long).unbind(-1)  # tensor containing all valid x- y- z- in grid_coords
        rospy.loginfo(f"No. of valid coords vs total grid coords: {valid_indices.sum().item()}/{grid_coords.shape[0]}")
        
        # Get the log odds of the occupancy for addition and minus operation
        occ_log_odds = torch.log(torch.div(self.voxel_grid[gx, gy, gz, self.voxel_occ_prob_layer], 1.0 - self.voxel_grid[gx, gy, gz, self.voxel_occ_prob_layer]))
        updated_occ_odds = self.update_occupancy_probability(occ_log_odds, occupancy_score_map, valid_indices)
        rospy.loginfo("Convert the log odds back to occupancy probabilities: odds=p/(1-p) => p=odds/(1+odds)")
        self.voxel_grid[gx, gy, gz, self.voxel_occ_prob_layer] = torch.div(updated_occ_odds, 1.0 + updated_occ_odds)
        
        self.update_semantic_class_and_probability(gx,gy,gz,valid_indices, semantics)
        self.voxel_grid[..., 1:3]=torch.clamp(self.voxel_grid[..., 1:3],self.eps,1.0 - self.eps) # clamping 1:occ_prob 2:sem_conf
        roi_coverage = self.get_ROI_evaluation_metric()
        return roi_coverage        

    def get_ROI_evaluation_metric(self):
        if self.evaluation_metric == 'semantic_class':
            voxel_layer = self.voxel_sem_cls_layer          # my own setting
            target_value = -1                               # class id are not -1
        if self.evaluation_metric == 'semantic_conf':       # Check the values within the target bounds and count the number of updated voxels
            voxel_layer = self.voxel_sem_conf_layer         # the same setting as GNBV
            target_value = 0.5                              # sem conf are not 0.5
        if self.evaluation_metric == 'occ_prob':       
            voxel_layer = self.voxel_occ_prob_layer         
            target_value = 0.5                              # sem occ prob are not 0.5
        
        if self.target_bounds is not None:
            target_voxels = self.voxel_grid[self.target_bounds[0] : self.target_bounds[3],self.target_bounds[1] : self.target_bounds[4],
                                            self.target_bounds[2] : self.target_bounds[5],voxel_layer]   
        explored = (target_voxels != target_value).sum()
        coverage = round(explored.item()/target_voxels.numel(), 6) 
        rospy.loginfo(f"Using evaluation metric: {self.evaluation_metric}. No of explored semantic voxels / total ROI voxels: {explored.item()}/{target_voxels.numel()}")
        
        # Update explored semantic voxels in ROI
        self.explored_semantic_voxels = self.explored_semantic_voxels_update
        self.explored_semantic_voxels_update = explored.item()
        rospy.loginfo(f"No. of explored semantic voxels updated from {self.explored_semantic_voxels} to {self.explored_semantic_voxels_update}")
        return coverage    
    
    def compute_mostion_cost(self, vp_posi, cam_posi):
        distance = torch.dist(vp_posi, cam_posi)            # Calculate the Euclidean distance
        mostion_cost = torch.exp(-1 * self.lamda * distance)                  # Compute e^(d)
        return mostion_cost
    
    def get_visible_semantics(self, vp_posi_idx, semantics_idx): 
        visible_semantics_idx = []
        for sem_voxel in semantics_idx:
            # print("sem_voxel is ", sem_voxel)
            traversed_voxels= raycast_3d(vp_posi_idx, sem_voxel)
            # print("traversed_voxels is ", traversed_voxels)
            if check_visibility(semantics_idx, traversed_voxels):
                # print(">> >> append this sem_voxel")
                visible_semantics_idx.append(sem_voxel)
        
        rospy.loginfo(f"Check visible semantic_voxels: no. of visible semantics is {len(visible_semantics_idx)}")
        return visible_semantics_idx

    def get_visible_semantics_new(self, vp_posi_idx, semantics_idx, occupied_indices): 
        visible_semantics_idx = []
        for sem_voxel in semantics_idx:
            # print("sem_voxel is ", sem_voxel)
            traversed_voxels = raycast_3d(vp_posi_idx, sem_voxel)
            # print("traversed_voxels is ", traversed_voxels)
            # if check_visibility(occupied_indices, traversed_voxels):
            if check_semantics_visibility_tensor(occupied_indices, traversed_voxels):
                # print(">> >> append this sem_voxel")
                visible_semantics_idx.append(sem_voxel)
        
        rospy.loginfo(f"Check visible semantic_voxels: no. of visible semantics is {len(visible_semantics_idx)}")
        return visible_semantics_idx    

    def compute_semantic_IG(self,sem_idx):
        """
        Compute semantic information gain
        :seman : voxel index 
        :return: semantic_IG
        """
        Ps = self.voxel_grid[sem_idx[0], sem_idx[1], sem_idx[2], self.voxel_sem_conf_layer]
        I_sem_x = -Ps * torch.log2(Ps) - (1 - Ps) * torch.log2(1 - Ps)      # Calculate I_sem(x)
        return I_sem_x

    def compute_expected_semantic_IG(self, visible_semantics_idx):
        G_sem = 0
        for sem_idx in visible_semantics_idx :
            I_sem_x = self.compute_semantic_IG(sem_idx)
            G_sem +=  I_sem_x
        return G_sem
    
    def compute_viewpoint_total_utility(self,visible_semantics_idx, vp_posi, cam_posi):
        G_sem = self.compute_expected_semantic_IG(visible_semantics_idx)
        mostion_cost = self.compute_mostion_cost(vp_posi, cam_posi)                  # Compute e^(d)
        total_utility = (G_sem * mostion_cost).item()
        rospy.loginfo(f"No. of visible semantics is {len(visible_semantics_idx)}, G_sem is {round(G_sem.item(),4)}, mostion cost is {round(mostion_cost.item(),4)}, total utility is {total_utility}") 
        return total_utility, G_sem, mostion_cost
    
    def viewpoint_evaluation(self, vp_Trans_w_o, optical_pose):
        rospy.logwarn("In my NBV, i calculated semantics in the FOV i did not constrain to the ROI")
        gx, gy, gz = self.raycasting(vp_Trans_w_o)
        # tuple(3) the occupied indices in current FOV
        occupied_indices = self.get_occupied_voxel_indexes(gx, gy, gz) 
        semantics_indices_fov = self.get_semantics_voxels_in_FOV(gx, gy, gz)
        
        vp_posi = vp_Trans_w_o[0, :3, 3]
        # Convert cam_pose to a PyTorch tensor and move to the same device as vp_posi
        cam_posi = torch.tensor(optical_pose[:3], device=vp_posi.device)
        
        # target in grid_coords
        vp_posi_idx_grid = torch.div(vp_posi - self.origin, self.voxel_size, rounding_mode="floor").to(torch.long).cpu().numpy() 
        semantics_idx_grid = semantics_indices_fov.cpu().numpy()
        
        if self.verbose_octomap:
            print("gx, gy, gz",gx, gy, gz)
            print("semantics_indices_fov",semantics_indices_fov)
            print("vp_posi grid_coords is :", vp_posi_idx_grid)
        
        # visible_semantics_idx = self.get_visible_semantics(vp_posi_idx_grid, semantics_idx_grid)
        visible_semantics_idx = self.get_visible_semantics_new(vp_posi_idx_grid, semantics_idx_grid, occupied_indices)
        
        total_utility, G_sem, mostion_cost = self.compute_viewpoint_total_utility(visible_semantics_idx, vp_posi, cam_posi)
        return total_utility, G_sem, mostion_cost
    
    def raycasting(self, vp_Trans_w_o):
        rospy.loginfo(f"Geting frustum points of the dummy viewpoint")
        (ray_origins,ray_directions,_) = get_frustum_points(None,    # not unit direction, it has length
            self.T_o_pc, vp_Trans_w_o,self.min_depth, self.max_depth, self.width, self.height, 
            self.intrinsics,self.occupied_lodds, self.free_lodds, self.device)
        # frustum 3D space: contain points from near to far and span 128 depth interval in world_frame
        ray_points = (ray_directions[:, :, None, :] * self.t_vals[None, :, None] + ray_origins[:, :, None, :]).view(-1, 3)
        
        # grid_coords: A tensor containing grid coordinates. Each row represents a 3D coords (x, y, z).
        grid_coords = torch.div(ray_points - self.origin, self.voxel_size, rounding_mode="floor")  
        valid_indices = self.get_valid_indices_from_octomap(grid_coords, self.voxel_dims)
        rospy.loginfo(f"No. of valid coords vs total grid coords: {valid_indices.sum().item()}/{grid_coords.shape[0]}")
        gx, gy, gz = grid_coords[valid_indices].to(torch.long).unbind(-1)
        return gx, gy, gz

    def get_semantics_voxels_in_FOV(self, gx, gy, gz):
        # Remove duplicated points: might be optional here
        # test_combined_indices = torch.stack((gx, gy, gz), dim=-1)
        # print("test_combined_indices are:",test_combined_indices, "size are: ",test_combined_indices.size())
        # unique_test_combined_indices = torch.unique(test_combined_indices, dim=0)
        # print("unique_test_combined_indices are:",unique_test_combined_indices, "size are: ",unique_test_combined_indices.size())
        # indexes from 28366080 to 331498
        
        current_class_id = self.voxel_grid[gx, gy, gz, self.voxel_sem_cls_layer]
        # Create a boolean mask for where current_class_id is greater than -1
        mask = current_class_id > self.init_sem_cls_id
        # mask_sum = mask.sum()
        # print("Sum of the mask:", mask_sum.item())            Compute the sum of the mask
        
        # Get indices where the current_class_id is greater than -1
        valid_semantic_id_indices = torch.nonzero(mask, as_tuple=True)
        filtered_gx = gx[valid_semantic_id_indices]
        filtered_gy = gy[valid_semantic_id_indices]
        filtered_gz = gz[valid_semantic_id_indices]
        
        # Optionally combine indices to get a list of tuples (if you need them in this format)
        combined_indices = torch.stack((filtered_gx, filtered_gy, filtered_gz), dim=-1)
        if self.verbose_octomap:
            rospy.loginfo(f"Indices in Octomap with non-background class: {combined_indices.shape[0]} elements found.")
        unique_combined_indices = torch.unique(combined_indices, dim=0)
        rospy.loginfo(f"After removing duplicated points, {unique_combined_indices.size()[0]} unique semantic voxels left.")
        return unique_combined_indices

    def get_occupied_voxel_indexes(self, gx, gy, gz):
        voxels_occ_prob = self.voxel_grid[gx, gy, gz, self.voxel_occ_prob_layer]
        mask = voxels_occ_prob > self.roi_occ_prob
        valid_occ_indices = torch.nonzero(mask, as_tuple=True)
        occ_gx = gx[valid_occ_indices]
        occ_gy = gy[valid_occ_indices]
        occ_gz = gz[valid_occ_indices]
        # Combine indices to get a list of tuples
        combined_indices = torch.stack((occ_gx, occ_gy, occ_gz), dim=-1)
        if self.verbose_octomap:
            rospy.loginfo(f"Occupied indices in Octomap: {combined_indices.shape[0]} elements found.")
        unique_combined_indices = torch.unique(combined_indices, dim=0)
        rospy.loginfo(f"Removing duplicated points: {len(occ_gz)} points becomes to {unique_combined_indices.size()[0]} unique occupied voxels left.")
        return unique_combined_indices 

    def set_my_target_roi_region(self, tar_posi: torch.tensor) -> None:
        """
        In MyNBV, the target ROI is defined identical to GNBV: a cube arond the target point
        :tar_posi: fruit position GT
        """ 
        rospy.loginfo("Target ROI is defined around the fruit position GT ")
        rospy.logwarn("You might need to update the set_my_target_roi() each time when you get new fruit pose or you can define larger target bounds to offset the effect of position changes.")
        if tar_posi is None:
            return
        target_coords = torch.div(tar_posi-self.origin, self.voxel_size, rounding_mode="floor").to(torch.long) # in grid_coords
        x_min = torch.clamp(target_coords[0] - self.roi_range[0], 0, self.voxel_dims[0])
        x_max = torch.clamp(target_coords[0] + self.roi_range[0], 0, self.voxel_dims[0])
        y_min = torch.clamp(target_coords[1] - self.roi_range[1], 0, self.voxel_dims[1])
        y_max = torch.clamp(target_coords[1] + self.roi_range[1], 0, self.voxel_dims[1])
        z_min = torch.clamp(target_coords[2] - self.roi_range[2], 0, self.voxel_dims[2])
        z_max = torch.clamp(target_coords[2] + self.roi_range[2], 0, self.voxel_dims[2])
        self.target_bounds = torch.tensor([x_min, y_min, z_min, x_max, y_max, z_max], device=self.device)  # ROI range in grid coords
        self.voxel_grid[x_min:x_max,y_min:y_max,z_min:z_max,self.voxel_roi_layer] = 1
        self.voxel_grid[x_min:x_max,y_min:y_max,z_min:z_max,self.voxel_sem_conf_layer] = self.roi_sem_conf # the same setting as GNBV
        self.voxel_grid[x_min:x_max,y_min:y_max,z_min:z_max,self.voxel_occ_prob_layer] = self.roi_occ_prob # my own adding
        self.voxel_grid[..., 1:3] = torch.clamp(self.voxel_grid[..., 1:3], self.eps, 1.0 - self.eps)   # clamping 1:occ_prob 2:sem_conf
        
        