import rospy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rospy
from scene_representation.my_raysampler import MyRaySampler
from scene_representation.my_raycast_algo_3d import raycast_3d, check_visibility,check_semantics_visibility_tensor
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation,get_frustum_points
from scene_representation.conversions import T_from_rot_trans_np
from scipy.spatial.transform import Rotation as R
from utils.py_utils import my_pause_program
import time

class MyVoxelGrid:
    """
    3D representation to store occupancy info & other features (e.g. semantics) over multiple viewpoints
    """
    def __init__(self,width: int,height: int,intrinsics: torch.tensor,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
        """
        :param width/height: image width/height
        :param intrinsics: camera intrinsics
        :param device: device to use for computation
        """
        self.device = device
        self.width = width
        self.height = height
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
        self.voxel_max_fusion_coef = rospy.get_param("voxel_max_fusion_coefficient", "0.9")             
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

    def get_occupied_points(self):
        """
        Returns the coordinates of the occupied points in the grid
        :return: tensor of shape (N, 3) containing the coordinates of the occupied points
        """
        grid_coords = torch.nonzero(self.voxel_grid[..., self.voxel_occ_prob_layer] > self.voxel_occupancy_threshold)
        points = grid_coords * self.voxel_size + self.origin
        # semantics_conf = self.voxel_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 2]
        class_ids = self.voxel_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 3]
        return points, class_ids
    
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
        
        # # Get the full voxel grid shape
        # voxel_shape = self.voxel_grid.shape
        
        # # Create a boolean mask for the entire voxel grid initialized to False
        # roi_mask = torch.zeros(voxel_shape, dtype=torch.bool)
        
        # # Set the ROI region in the mask to True (where the ROI is)
        # roi_mask[self.target_bounds[0]:self.target_bounds[3],
        #         self.target_bounds[1]:self.target_bounds[4],
        #         self.target_bounds[2]:self.target_bounds[5]] = True
        
        # grid_coords = torch.nonzero(self.voxel_grid[..., self.voxel_occ_prob_layer] > self.voxel_occupancy_threshold)
        # non_roi_mask = ~roi_mask
        # non_roi_points = torch.nonzero(non_roi_mask & (self.voxel_grid[..., self.voxel_occ_prob_layer] > 0))

        
        # points = grid_coords * self.voxel_size + self.origin
        # semantics_conf = self.voxel_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 2]
        # class_ids = self.voxel_grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2], 3]
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
        # log_odds = log(p / (1 - p)): 1 is occ_prob, 2 is sem_conf  | 1:3 means two layers
        # occ_log_odds should be all 0 at the beginning
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
        #TODO: Set the ROI of these voxels to 1  (I may not need this 2 lines)
        # Utility_score: E[target_voxels]
        # valid_class_mask = (self.voxel_grid[..., self.voxel_sem_cls_layer] >=0)
        # self.voxel_grid[..., self.voxel_roi_layer][valid_class_mask] = 1
        
        Ps = self.voxel_grid[sem_idx[0], sem_idx[1], sem_idx[2], self.voxel_sem_conf_layer]
        I_sem_x = -Ps * torch.log2(Ps) - (1 - Ps) * torch.log2(1 - Ps)      # Calculate I_sem(x)
        return I_sem_x

    def compute_expected_semantic_IG(self, visible_semantics_idx):
        # Usem = Gsem(v) * e^d; G_sem(v): the expected semantic information gain 
        G_sem = 0
        for sem_idx in visible_semantics_idx :
            I_sem_x = self.compute_semantic_IG(sem_idx)
            G_sem +=  I_sem_x
            # print("I_sem_x is ", I_sem_x) 
        return G_sem
    
    def compute_viewpoint_total_utility(self,visible_semantics_idx, vp_posi, cam_posi):
        G_sem = self.compute_expected_semantic_IG(visible_semantics_idx)
        mostion_cost = self.compute_mostion_cost(vp_posi, cam_posi)                  # Compute e^(d)
        total_utility = (G_sem * mostion_cost).item()
        rospy.loginfo(f"No. of visible semantics is {len(visible_semantics_idx)}, G_sem is {round(G_sem.item(),4)}, mostion cost is {round(mostion_cost.item(),4)}, total utility is {total_utility}") 
        return total_utility, G_sem, mostion_cost
    
    def viewpoint_evaluation(self, vp_Trans_w_o, optical_pose):
        rospy.logwarn("In my NBV, i calculated semantics in the FOV i did not constrain to the ROI")
        
        # start = time.time()
        
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
        # print("1 raycasting: ", time.time()-start)
        # my_pause_program()
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
        
    












if __name__ == "__main__":
    # Create a ray sampler
    tensor_c = torch.rand(10, 3)
    # Define a NumPy array with a very small value
    array_d = np.array([0.003])

    # Convert NumPy array to a PyTorch tensor
    tensor_d = torch.tensor(array_d, dtype=torch.float32)

    # Perform element-wise division
    result = torch.div(tensor_c, tensor_d, rounding_mode="floor")
    
    print(result)
    voxel_dims=torch.tensor(np.array([100,250,105]), dtype=torch.float32)
    valid_indices = get_valid_indices(result, voxel_dims)
    print(valid_indices)
    
    gx, gy, gz = result[valid_indices].to(torch.long).unbind(-1)
    print(gx, gy, gz)
    
    if False:
        print("Test case 2")
        # Assuming `voxel_grid` and `num_features` are defined as:
        num_features = 4  # 0: ROI, 1: occ_prob, 2: sem_conf, 3: sem_class_id
        voxel_grid = torch.zeros(4, 4, 3, num_features, dtype=torch.float32, device='cuda')  # Example device

        # Set up some example values for `sem_class_id`
        # For demonstration, randomly assign some class IDs (some within 0 to 3, some outside)
        torch.manual_seed(0)
        voxel_grid[..., 3] = torch.randint(0, 10, (4, 4, 3))  # random integers from 0 to 4

        # Check which voxels have `sem_class_id` within the range [0, 3]
        # valid_class_mask = (voxel_grid[..., 3] >= 0) & (voxel_grid[..., 3] <= 3)
        valid_class_mask = (voxel_grid[..., 3] >=9)
        
        # Set the ROI of these voxels to 1
        voxel_grid[..., 0][valid_class_mask] = 1

        # Print to verify changes
        print("Voxel grid updates (showing only a few slices for brevity):")
        print("sem_class_ids:")
        # print(voxel_grid[:, :, :, 3][0])  # Display some sem_class_id values from the first slice
        print(voxel_grid[:, :, :, 3])  # Display some sem_class_id values from the first slice
        print(valid_class_mask)  # Display some sem_class_id values from the first slice
        print("ROIs set based on sem_class_ids:")
        # print(voxel_grid[:, :, :, 0][0])  #
        print(voxel_grid[:, :, :, 0])  #
    
    