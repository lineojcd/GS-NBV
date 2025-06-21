import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
from scene_representation.conversions import T_from_rot_trans_np
import rospy

def look_at_rotation(
    eye: torch.tensor,
    target: torch.tensor,
    ref: torch.tensor = torch.tensor([1.0, 0.0, 0.0]),
    up: torch.tensor = torch.tensor([0.0, 0.0, -1.0]),
) -> torch.tensor:
    """
    Compute the quaternion rotation to look at a target from a given eye position
    :param eye: eye position
    :param target: target position
    :param ref: reference vector
    :return: quaternion rotation
    """
    dir = target - eye
    dir = dir / torch.norm(dir)
    ref = ref.to(dir.device).to(dir.dtype)
    up = up.to(dir.device).to(dir.dtype)
    one = torch.ones(1, device=dir.device, dtype=dir.dtype)
    # Calculate quaternion between reference vector and direction vector
    vec1 = torch.cross(ref, dir)
    w1 = one + torch.dot(ref, dir)
    quat1 = torch.cat((w1, vec1))
    quat1 = quat1 / torch.norm(quat1)
    return quat1


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :return: tensor of rotation matrices (..., 3, 3)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def transform_from_rotation_translation(
    quaternions: torch.Tensor, translations: torch.Tensor
) -> torch.Tensor:
    """
    Convert rotations given as quaternions and translation vectors to 4x4 transformation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :param translations: tensor of translation vectors (..., 3)
    :return: tensor of transformation matrices (..., 4, 4)
    """
    # print("quaternions.shape[0]:",quaternions.shape[0])    # 1
    matrices = (
        torch.eye(4, device=quaternions.device)
        .unsqueeze(0)
        .repeat(quaternions.shape[0], 1, 1)
    )
    matrices[:, :3, :3] = quaternion_to_matrix(quaternions)
    matrices[:, :3, 3] = translations
    return matrices

def transform_from_rot_trans(
    rotation: torch.Tensor, translations: torch.Tensor
) -> torch.Tensor:
    """
    Convert rotations given as quaternions and translation vectors to 4x4 transformation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :param translations: tensor of translation vectors (..., 3)
    :return: tensor of transformation matrices (..., 4, 4)
    """
    # print("rotation.shape[0]:",rotation.shape[0])
    matrices = (
        torch.eye(4, device=rotation.device)
        .unsqueeze(0)
        .repeat(rotation.shape[0], 1, 1)
    )
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = translations
    return matrices


def generate_pixel_coords(width=640, height=640, device = torch.device("cuda:0"), shift_pixel=True):    
    """
    Create a mesh grid of (u, v) coordinates
    :return pixel_coords: [u, v, 1] shape; ([640, 640, 3])
    """
    u, v = torch.meshgrid(          #u:width v:height
        [   torch.arange(0.0, width, device=device, dtype=torch.float32),
            torch.arange(0.0, height, device=device, dtype=torch.float32),
        ],  indexing="xy",
    )
    if shift_pixel:
        # the pixel coordinates are shifted to the center of each pixel
        u, v = u + 0.5, v + 0.5
    # Convert the pixel coordinates to homogeneous coordinates:  [u, v, 1]
    pixel_coords = torch.stack((u, v, torch.ones_like(u)), dim=-1)
    # print("pixel_coords",pixel_coords.shape)
    # print("pixel_coords",pixel_coords)
    return pixel_coords

def get_centroid(tensor_cpu, filter_fruit_noise, std_level, verbose = False):
    # Another simple algo, did not used here
    if filter_fruit_noise:
        print("filter noisy fruit points to get centroid")
        # Step 1: Calculate the mean and std for the points
        mean = tensor_cpu.mean(dim=1)
        std = tensor_cpu.std(dim=1, keepdim=True)
        
        # Find points that lie within 1 standard deviation for each coordinate (x, y, z)
        within_std_mask = (tensor_cpu >= (mean - std)) & (tensor_cpu <= (mean + std))

        # Keep only the points that satisfy the condition for all three coordinates
        filtered_points = tensor_cpu[within_std_mask.all(dim=2)]

        # # Step 2: Calculate the Euclidean distances of each point from the mean
        # distances = torch.norm(tensor_cpu - mean_position.unsqueeze(1), dim=2)
        # # Step 3: Calculate the standard deviation of these distances
        # std_dev = distances.std()
        # # Step 4: Filter out points within 2 standard deviations
        # mask = distances <= std_level * std_dev
        # filtered_points = tensor_cpu[mask].reshape(-1, 3)
        
        # print("filtered_points", filtered_points)
        
        # Step 5: Recalculate the mean using the filtered points
        filtered_mean = filtered_points.mean(dim=0)
        unfilter_mean = tensor_cpu.mean(dim=1)
        
        if verbose:
            print("tensor_cpu shape", tensor_cpu.shape)
            print("filtered_points shape", filtered_points.shape)
        
        print("Unfilter result is:", unfilter_mean, " ; Filtered result is", filtered_mean )
        return filtered_mean
    else:
        return tensor_cpu.mean(dim=1)

def transform_points(points: torch.tensor,transforms: torch.tensor) -> torch.tensor:
    """
    Transform a point cloud from 'frame a ' to 'frame b'
    :param points: point cloud
    :param transforms: transformation matrices
    """
    points_h = nn.functional.pad(points, (0, 1), "constant", 1.0) # pad 1 to the vector: from [ a,b,c] to [a,b,c,1]
    points_w = points_h @ transforms.permute(0, 2, 1)  # permute conducted a transpose here
    return points_w[:, :, :3]

def transform_points_my(points: torch.tensor,transforms: torch.tensor) -> torch.tensor:
    """
    Transform a point cloud from 'frame a ' to 'frame b'
    :param points: point cloud
    :param transforms: transformation matrices
    """
    points_h = nn.functional.pad(points, (0, 1), "constant", 1.0) # pad 1 to the vector: from [ a,b,c] to [a,b,c,1]
    # print("transforms.shape",transforms.shape)
    # print("transforms\n",transforms)
    # n * 4 x 4 * 4 = n * 4
    points_w = points_h @ transforms.permute(0, 2, 1)  # permute conducted a transpose here
    # remove the last 1
    return points_w[:, :, :3]

def transform_points_cam_to_world_my(
    points: torch.tensor,T_w_o: torch.tensor, arm_type='gnbv', device = torch.device("cuda:0")) -> torch.tensor:
        """
        Transform a point cloud from 'camera_frame' to 'world_frame'
        :param points: point cloud
        :param T_w_o: transformation matrices
        """
        # TODO: remove this hack, only for gazebo; because the initial Z-coords of arm is 0.024
        # print("points.shape:",points.shape)
        if arm_type == 'gnbv':
            points[..., 1] += 0.024     # camera coords: [x,y,z] y in cam -> z in world
        else:
            rospy.logwarn("Delete this at the end: using jaco2 arm")
            # print("points[..., 1]",points[..., 1])
        
        # Transformation from pixel camera frame to optical frame
        # r_my = R.from_quat([0.5, -0.5, 0.5, 0.5])
        # r_my = R.from_quat([0.33333333, 0.0, 0.66666667, 0.66666667])
        # r_my = np.array([[0., -1., 0.],
        #                 [1., 0., 0.],
        #                 [1., 0., 1.]])
        r_my = np.array([[0., -1., 0.],         #Q form: 0  , 0  , 0.7071, 0.7071
                        [1., 0., 0.],
                        [0., 0., 1.]])
        T_o_pc_my = T_from_rot_trans_np(r_my, np.zeros((1, 3)))    # 4x4 last row 0 0 0 1
        T_o_pc = torch.as_tensor(T_o_pc_my, dtype=torch.float32, device=device)
        T_w_pc = T_w_o.to(torch.float32) @ T_o_pc                   
        return transform_points_my(points, T_w_pc)

def transform_points_cam_to_world_gnbv(
    points: torch.tensor,
    transforms: torch.tensor, 
    arm_type='gnbv', 
    device = torch.device("cuda:0")
    ) -> torch.tensor:
        """
        Transform a point cloud from 'camera_frame' to 'world_frame'
        :param points: point cloud
        :param transforms: transformation matrices
        """
        # TODO: remove this hack, only for gazebo; because the initial Z-coords of arm is 0.024
        if arm_type == 'gnbv':
            # camera coords: [x,y,z] y in cam -> z in world
            points[..., 1] += 0.024  
            pass
        else:
            pass
        
        # Transformation from optical frame to camera frame
        r = R.from_euler("xyz", [-np.pi / 2, 0.0, -np.pi / 2])
        T_oc = T_from_rot_trans_np(r.as_matrix(), np.zeros((1, 3)))
        T_oc = torch.as_tensor(T_oc, dtype=torch.float32, device=device)
        
        T_cws = transforms.to(torch.float32)
        T_ows = T_cws @ T_oc                   # T_oc = self.T_oc.clone()
        return transform_points(points, T_ows)

def get_frustum_points(
    depth_img:torch.tensor, 
    T_o_pc: torch.tensor, 
    transforms: torch.tensor, 
    min_depth: float, 
    max_depth:float,
    width: int,
    height: int,
    intrinsic: torch.tensor, 
    occupied_lodds:float,
    free_lodds:float,
    device = torch.device("cuda:0")
    ):
    """
    Compute the origins and directions for all rays in world frame
    :param depth_image: depth image (batch_size x width x height)
    :param T_oc: optical frame to camera frame transform
    :param transforms: transformation matrices (batch_size x 4 x 4)
    :return: ray origins, directions, occupancy_score_mask
    """
    batch_size = transforms.shape[0]  # torch.Size([1, 4, 4])  ; batch size: 1 
    
    # Set min and max depth: z_near = 0.05, z_far = 0.72,   
    min_depths = min_depth* torch.ones((batch_size, width * height),dtype=torch.float32,device=device) 
    
    # If depth image is provided, use it to compute the max depth
    if depth_img is not None:
        # Fill the nan value by the farest depth values
        depth_img[torch.isnan(depth_img)] = max_depth
        # get the max_depth
        max_depths = depth_img.view(1, -1)
    else:
        # Otherwise, use the far clipping plane
        max_depths = max_depth * torch.ones((batch_size, width * height),dtype=torch.float32,device=device)
    
    # Create a mask that is log odds 0.9 if the depth is less than far and log odds of 0.2 otherwise
    occupancy_score_mask = torch.where(max_depths < max_depth, occupied_lodds, free_lodds)
    
    # Convert the pixel coordinates to homogeneous coordinates:  [u, v, 1]
    pixel_coords = generate_pixel_coords(width, height, device)
    # Transform the camera coordinates to world coordinates: camera coordinates: [u~, v~, 1]
    camera_coords = pixel_coords.view(-1, 3) @ torch.inverse(intrinsic).t().type(torch.float32)
    
    # closest and furthest depth range
    ray_origins = (camera_coords * min_depths.unsqueeze(-1)).view(batch_size, -1, 3)
    ray_targets = (camera_coords * max_depths.unsqueeze(-1)).view(batch_size, -1, 3)
    
    # T_cws = transforms.to(torch.float32)
    # transforms = T_cws @ T_oc                                           # T_ows 
    
    T_w_o = transforms.to(torch.float32)
    T_w_pc = T_w_o @ T_o_pc  
    
    
    # Transform a point cloud from 'camera_frame' to 'world_frame'
    # ray_origins = transform_points_my(ray_origins, transforms)
    # ray_targets = transform_points_my(ray_targets, transforms)
    
    ray_origins = transform_points_my(ray_origins, T_w_pc)
    ray_targets = transform_points_my(ray_targets, T_w_pc)
    
    # Compute the ray directions
    ray_directions = ray_targets - ray_origins
    return ray_origins,ray_directions,occupancy_score_mask