"""
Author: Akshay K. Burusa
Maintainer: Akshay K. Burusa
"""

import torch
import rospy
import torch.nn as nn
import numpy as np
import open3d as o3d

from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as scipy_r

from scene_representation.conversions import T_from_rot_trans_np, T_from_rot_trans


class MyRaySampler:
    def __init__(self, width: int, height: int, intrinsic: torch.tensor, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        Constructor
        :param width/height: image width/height
        :param device: device to use for computation
        """
        self.nbv_method = 'gnbv'
        # self.nbv_method = 'mynbv'
        self.width = width
        self.height = height
        self.z_near = rospy.get_param("near_clipping_plane", "0.03")
        self.z_far = rospy.get_param("far_clipping_plane", "0.72")
        self.free_odds = -1.4
        self.occ_odds = 2.2
        # self.intrinsic = torch.tensor(
        #     [   [fx, 0, cx],      fx, fy: focal length along x-axis, y-axis, float number 
        #         [0, fy, cy],      cx, cy: principal point along x-axis, y-axis, float number     
        #         [0, 0, 1], ],
        #     dtype=torch.float32,device=device)        
        self.intrinsic = intrinsic
        self.device = device
        
        # Generate camera coordinates, which will be used for ray sampling
        self.generate_camera_coords()
        
        # Transformation from optical frame to camera frame
        r = scipy_r.from_euler("xyz", [-np.pi / 2, 0.0, -np.pi / 2])
        self.T_oc = T_from_rot_trans_np(r.as_matrix(), np.zeros((1, 3)))
        self.T_oc = torch.as_tensor(self.T_oc, dtype=torch.float32, device=self.device)

    def generate_camera_coords(self):
        """
        Generate camera coordinates, which will be used for ray sampling
        """
        # Create a mesh grid of (u, v) coordinates
        u, v = torch.meshgrid(
            [   torch.arange(0.0, self.width, device=self.device, dtype=torch.float32),
                torch.arange(0.0, self.height, device=self.device, dtype=torch.float32),
            ],
            indexing="xy",
        )
        # the pixel coordinates are shifted to the center of each pixel
        u, v = u + 0.5, v + 0.5
        
        # Convert the pixel coordinates to homogeneous coordinates:  [u, v, 1]
        pixel_coords = torch.stack((u, v, torch.ones_like(u)), dim=-1)
        
        # Transform the pixel coordinates to camera coordinates: [x, y, 1]
        # Why transpose here? I do not understand
        self.camera_coords = pixel_coords.view(-1, 3) @ torch.inverse(self.intrinsic).t().type(torch.float32)
        # print("camera_coords shape is ",self.camera_coords.shape) # ([409600, 3])
        
    def get_camera_coords(self):
        return self.camera_coords.clone()
        
    def ray_origins_directions(
        self,
        transforms: torch.tensor,
        depth_image: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute the origins and directions for all rays
        :param depth_image: depth image (batch_size x width x height)
        :param transforms: transformation matrices (batch_size x 4 x 4)
        :return: ray origins and directions and occupancy_score_mask
        """
        batch_size = transforms.shape[0]  # torch.Size([1, 4, 4])  ; batch size: 1
        # print("transforms.shape is ",transforms.shape)
        
        # Set min depth
        # z_near = 0.05, z_far = 0.72,
        min_depths = self.z_near * torch.ones((batch_size, self.width * self.height),dtype=torch.float32,device=self.device) 
        
        # Set Max depth
        # If depth image is provided, use it to compute the max depth
        if depth_image is not None:
            # Fill the nan value by the farest depth values
            depth_image[torch.isnan(depth_image)] = self.z_far
            # get the max_depth
            max_depths = depth_image.view(1, -1)
        else:
        # Otherwise, use the far clipping plane
            max_depths = self.z_far * torch.ones(
                (batch_size, self.width * self.height),
                dtype=torch.float32,
                device=self.device,
            )
            
        # Create a mask that is log odds 0.9 if the depth is less than far and log odds of 0.4 otherwise
        # log_odds(0.9)  # 2.2 occupied_odds | log_odds(0.4)  # -0.4 free | log_odds(0.5)  # 0 | log_odds(0.2) = -1.4
        # points_mask = torch.where(max_depths < self.z_far, 2.2, -0.4)
        occupancy_score_mask = torch.where(max_depths < self.z_far, self.occ_odds, self.free_odds)
        
        # Transform the camera coordinates to world coordinates: camera coordinates: [u~, v~, 1]
        camera_coords = self.camera_coords.clone()
        
        # close depth range
        ray_origins = (camera_coords * min_depths.unsqueeze(-1)).view(batch_size, -1, 3)
        # furthest depth range
        ray_targets = (camera_coords * max_depths.unsqueeze(-1)).view(batch_size, -1, 3)
        
        # Transform a point cloud from 'camera_frame' to 'world_frame'
        ray_origins = self.transform_points(ray_origins, transforms)
        ray_targets = self.transform_points(ray_targets, transforms)
        # Compute the ray directions
        ray_directions = ray_targets - ray_origins
        
        # print("ray_origins shape:", ray_origins.shape) # should be sth like [1, w*h, 3]
        # print("ray_targets shape:", ray_targets.shape)
        # print("ray_directions shape:", ray_directions.shape)
        return ray_origins, ray_directions, occupancy_score_mask

    def transform_points(self,points: torch.tensor,transforms: torch.tensor) -> torch.tensor:
        """
        Transform a point cloud from 'camera_frame' to 'world_frame'
        :param points: point cloud
        :param transforms: transformation matrices
        """
        # TODO: remove this hack, only for gazebo; because the initial Z-coords of arm is 0.024
        if self.nbv_method == 'gnbv':
            # camera coords: [x,y,z] y in cam -> z in world
            points[..., 1] += 0.024  
        else:
            pass
        
        # T_oc = self.T_oc.clone().requires_grad_()
        T_oc = self.T_oc.clone().requires_grad_()
        # T_cws = transforms.clone().to(torch.float32).requires_grad_()
        T_cws = transforms.clone().to(torch.float32).requires_grad_()
        T_ows = T_cws @ T_oc
        # pad 1 to the vector: from [ a,b,c] to [a,b,c,1]
        points_h = nn.functional.pad(points, (0, 1), "constant", 1.0)
        points_w = points_h @ T_ows.permute(0, 2, 1)
        return points_w[:, :, :3]






if __name__ == "__main__":
    print("test mode")
    # Create a ray sampler
    width=150
    height=150
    width=640
    height=640
    z_near=0.00
    z_near=0.05
    cx=0
    cy=0
    sampler = MyRaySampler(
        # width=640,
        # height=480,
        width=width,
        height=height,
        fx=824.2428421710242,
        fy=824.2428421710242,
        cx=300.5,
        cy=225.5,
        # cx=cx,
        # cy=cy,
        # z_near=0.1,
        z_near=z_near,
        z_far=1.0,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Camera coordinates:", sampler.camera_coords.shape)

    # Transformation matrix: identity
    T = np.eye(4).reshape(1, 4, 4)
    transforms = torch.tensor(T, dtype=torch.float32, device=sampler.device)
    print("The transforms is :", transforms)
    
    random_depth_image = torch.ones((1, width, height), dtype=torch.float32, device=sampler.device)
    # Create a depth_image with random values
    # random_depth_image = torch.rand((1, width, height), dtype=torch.float32, device=device)
    
    # print("random_depth_image:", random_depth_image)
    # Find the minimum and maximum values in the tensor
    min_value = torch.min(random_depth_image)
    max_value = torch.max(random_depth_image)
    # Print the minimum and maximum values
    # print(f"Minimum value in the tensor: {min_value.item()}")  # should be inside 0 - 1
    # print(f"Maximum value in the tensor: {max_value.item()}")
    
    
    
    # Compute the ray origins and directions
    ray_origins, ray_directions, points_mask = sampler.ray_origins_directions(
        transforms, random_depth_image
    )
    
    print("ray_origins is :",ray_origins)
    # Print the first two samples from the tensor
    first_sample = ray_origins[:, 0, :]
    second_sample = ray_origins[:, 1, :]
    last_second_sample = ray_origins[:, -2, :]  # Second to last sample
    last_sample = ray_origins[:, -1, :]         # Last sample
    print("First sample:", first_sample)        #[0.1000, 0.0364, 0.0033] : depth x width x height
    print("Second sample:", second_sample)      #[0.1000, 0.0363, 0.0033]
    print("Second to last sample:", last_second_sample.squeeze())   # [ 0.1000,  0.0124, -0.0148]
    print("Last sample:", last_sample.squeeze()) #[ 0.1000,  0.0123, -0.0148]
    print("ray_directions is :",ray_directions)
    print("points_mask is :",points_mask)
    # min_value = torch.min(points_mask)
    # max_value = torch.max(points_mask)
    # Print the minimum and maximum values
    # print(f"Minimum value in the tensor: {min_value.item()}") # should between: -0.4 to 2.2
    # print(f"Maximum value in the tensor: {max_value.item()}")
    
    # t_vals = torch.linspace(0.0, 1.0, 10, dtype=torch.float32, device=device)
    t_vals = torch.linspace(0.0, 1.0, 128, dtype=torch.float32, device=device)
    
    ray_points = (
        ray_directions[:, :, None, :] * t_vals[None, :, None]
        + ray_origins[:, :, None, :]
    ).view(-1, 3)
    
    print("ray_points shape:", ray_points.shape)
    first_sample = ray_points[0, :]
    second_sample = ray_points[ 1, :]
    last_second_sample = ray_points[ -2, :]  # Second to last sample
    last_sample = ray_points[ -1, :]         # Last sample
    print("First sample:", first_sample)        #[0.1000, 0.0364, 0.0033] : depth x width x height
    print("Second sample:", second_sample)      #[0.1000, 0.0363, 0.0033]
    print("Second to last sample:", last_second_sample.squeeze())   # [ 0.1000,  0.0124, -0.0148]
    print("Last sample:", last_sample.squeeze())
    # ray_points shape: torch.Size([3840000, 3]) = 128 * 3

    # Visualize the camera coordinates in Open3D
    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(
    #     ray_points.detach().cpu().numpy().astype(np.float64)
    # )
    # o3d.visualization.draw_geometries([origin_frame, pcd])
    
    
    show = False
    show = True
    if show:
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ray_points.detach().cpu().numpy().astype(np.float64))
        o3d.visualization.draw_geometries([origin_frame, pcd])
