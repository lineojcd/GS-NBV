import numpy as np
from viewpoint_planners.my_method_iter import *
from utils.py_utils import look_at_rotation
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation
import rospy
from numpy.random import default_rng

rng = default_rng()

class SCViewpointSampler:
    """
    Generate viewpoint samples for a given scene according to the given constraints
    """
    def __init__(self, seed: int = 2024):
        """
        Initialize the viewpoint sampler
        :param num_samples: Number of viewpoints to sample
        """
        self.num_samples = rospy.get_param("SCNBV_num_samples", "50")
        self.view_samples = np.empty((0, 7))
        self.seed = seed
        # np.random.seed(self.seed)
        self.pts_dict = None
        self.arm_max_reach = rospy.get_param("arm_max_reach", "0.60")
        self.arm_min_reach = rospy.get_param("arm_min_reach", "0.10")
        self.vp_sampling_spherical_R = rospy.get_param("SCNBV_sampling_spherical_R", "0.1")
        self.vp_sampling_Rmin = rospy.get_param("SCNBV_sampling_Rmin", "0.21")
        self.vp_sampling_Rmax = rospy.get_param("SCNBV_sampling_Rmax", "0.26")
        self.vp_sampling_phi = np.radians(rospy.get_param("SCNBV_sampling_phi", "180"))  # degree to radian
        self.show_plot = False 
        self.verbose_vpsampler = rospy.get_param("verbose_vpsampler")
        self.arm_origin = np.array([float(x) for x in rospy.get_param('arm_origin').split()])
        self.workspace_bound_a = rospy.get_param("SCNBV_workspace_bound_left", "225")
        self.workspace_bound_b = rospy.get_param("SCNBV_workspace_bound_right", "315")
        print("show me what is the bound", self.workspace_bound_a, self.workspace_bound_b)
        self.num_theta = 360 - self.workspace_bound_b  + self.workspace_bound_a
        self.num_phi = rospy.get_param("SCNBV_sampling_phi", "180")
        
        
    def SCNBVP_sampler_init():
        phi = np.linspace(0, np.pi, self.num_phi)                   # sample 180 pts from 0 to pi/2 
        theta = np.linspace(-0.25 * np.pi, 1.25 * np.pi, self.num_theta)  # theta from -0.25*pi to 1.25*pi
        phi, theta = np.meshgrid(phi, theta)                        # Meshgrid for proper parameterization
        # Radii for the spheres
        radii = np.arange(self.vp_sampling_Rmin, self.vp_sampling_Rmax, 0.01)  # radii from 0.21 to 0.26 with step size 0.01
            
    def SCNBVP_spherical_sampler(self, v_last, radius, num):
        theta = np.random.uniform(0, 2*np.pi, num)  
        if self.verbose_vpsampler:
            print("thetha min and max are ", np.min(theta), np.max(theta))
        phi = np.random.uniform(0, self.vp_sampling_phi, num)  # Random azimuthal angles (0 to 2pi)
        x = radius * np.sin(phi) * np.cos(theta)    # Convert spherical coordinates to Cartesian coordinates
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points = np.stack((x, y, z), axis=-1)
        if self.verbose_vpsampler:
            print("v_last", v_last)
        points_world = points + v_last
        reachable_points = self.arm_reachability_filter(points_world)     # Arm reachable check
        return reachable_points
            
    def SCNBVP_spherical_wedge_sampler(self, Q_est, radius, num):
        ws_bd_a_rad = self.degree_to_radian(self.workspace_bound_a)
        ws_bd_b_rad = self.degree_to_radian(self.workspace_bound_b)
        theta = np.random.uniform(ws_bd_a_rad, ws_bd_b_rad, num)  
        phi = np.random.uniform(0, self.vp_sampling_phi, num)  # Random azimuthal angles (0 to 2pi)
        x = radius * np.sin(phi) * np.cos(theta)    # Convert spherical coordinates to Cartesian coordinates
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points = np.stack((x, y, z), axis=-1)
        # points_world = points + Q_est.squeeze().numpy() 
        points_world = points + Q_est
        points_world = np.array(points_world)
        if self.verbose_vpsampler:
            print("thetha min and max are ", np.min(theta), np.max(theta))
            print("points:", points)
            print("points_world:", points_world)
            # print("Q_est:", Q_est.squeeze().numpy())
            print("Q_est:", Q_est)
            print("points_world shape:", points_world.shape)
            
        reachable_points = self.arm_reachability_filter(points_world)     # Arm reachable check
        return reachable_points
    
    def SCNBVP_sampler_3D_world(self, Q_est, optical_pose):
        num = 1000
        n = self.num_samples
        r = np.random.uniform(self.vp_sampling_Rmin, self.vp_sampling_Rmax, num)  
        reachable_points = self.SCNBVP_spherical_wedge_sampler(Q_est, r, num)
        num_of_valid_pts = len(reachable_points)
        print("get num_of_valid_pts", num_of_valid_pts)
        # make sure n/2 = 25 points
        if num_of_valid_pts > n/2:
            rospy.logwarn("Sampling n/2 global points")
            random_points_idx = np.random.randint(0, num_of_valid_pts, int(n/2))
            reachable_points = np.array(reachable_points)
            valid_vp_list =  reachable_points[random_points_idx].tolist()
        else:
            rospy.logwarn(f'Less than n/2 global sampling points: {num_of_valid_pts}')
        
        # Local Sampling
        v_last = optical_pose[:3]
        r_local = np.random.uniform(self.vp_sampling_spherical_R, self.vp_sampling_spherical_R, num)  
        reachable_points_local = self.SCNBVP_spherical_sampler(v_last, r_local, num)
        num_of_valid_pts_local = len(reachable_points_local)
        print("get num_of_valid_pts_local", num_of_valid_pts_local)
        
        if num_of_valid_pts_local > n/2:
            rospy.logwarn("Sampling n/2 local points")
            local_random_points_idx = np.random.randint(0, num_of_valid_pts_local, int(n/2))
            reachable_points_local = np.array(reachable_points_local)
            local_valid_vp_list =  reachable_points_local[local_random_points_idx].tolist()
        else:
            rospy.logwarn(f'Less than n/2 local sampling points: {num_of_valid_pts_local}')
        
        valid_vp_list.extend(local_valid_vp_list)
        return valid_vp_list
        
    def SCNBVP_sampler_2D_world(self, Q_est, optical_pose):
        num = 1000
        n = self.num_samples
        r = np.random.uniform(self.vp_sampling_Rmin, self.vp_sampling_Rmin, num)  
        reachable_points = self.SCNBVP_spherical_wedge_sampler(Q_est, r, num)
        num_of_valid_pts = len(reachable_points)
        
        # make sure n/2 = 25 points
        if num_of_valid_pts > n/2:
            rospy.logwarn("Sampling n/2 global points")
            random_points_idx = np.random.randint(0, num_of_valid_pts, n/2)
            reachable_points = np.array(reachable_points)
            valid_vp_list =  reachable_points[random_points_idx].tolist()
        else:
            rospy.logwarn("Less than n/2 global sampling points")
        
        # Local Sampling
        v_last = optical_pose[:3]
        r_local = np.random.uniform(self.vp_sampling_spherical_R, self.vp_sampling_spherical_R, num)  
        reachable_points_local = self.SCNBVP_spherical_sampler(v_last, r_local, num)
        num_of_valid_pts_local = len(reachable_points_local)
        
        if num_of_valid_pts_local > n/2:
            rospy.logwarn("Sampling n/2 local points")
            local_random_points_idx = np.random.randint(0, num_of_valid_pts_local, n/2)
            reachable_points_local = np.array(reachable_points_local)
            local_valid_vp_list =  reachable_points_local[local_random_points_idx].tolist()
        else:
            rospy.logwarn("Less than n/2 local sampling points")
        
        valid_vp_list.extend(local_valid_vp_list)
        return valid_vp_list
        
        
        
        
        
        
        
    def arm_reachability_filter(self, valid_viewpoint_list):
        # Filter the valid_viewpoint_list by arm reachability
        return [vp for vp in valid_viewpoint_list 
                if self.arm_min_reach <= np.linalg.norm(vp - self.arm_origin) <= self.arm_max_reach]

    def get_auxiliary_vector(self, axis):
        # Create an orthogonal vector to axis
        if (axis == np.array([1, 0, 0])).all():
            first_vector = np.cross(axis, np.array([0, 1, 0]))
        else:
            first_vector = np.cross(axis, np.array([1, 0, 0]))
        first_vector = first_vector / np.linalg.norm(first_vector)
        second_vector = np.cross(axis, first_vector)
        second_vector = -1* second_vector / np.linalg.norm(second_vector) +np.array([0, 0, 0])
        if self.verbose_vpsampler:
            print("orthogonal_vector and  second_vector are", first_vector, second_vector)
        return first_vector, second_vector
    
    def get_vp_from_ring(self, samples=360):
        axis = np.array(self.f_axis)
        axis = axis / np.linalg.norm(axis)            # Normalize the axis vector
        # Create auxiliary vectors to axis
        first_vector, second_vector = self.get_auxiliary_vector(axis)
        
        # Generate points on the circle
        # theta = np.linspace(0, 2* np.pi, samples - 1, endpoint=True)  # Ensures points are evenly spaced
        theta = np.linspace(0, 2* np.pi, samples -1 , endpoint=False)  # Ensures points are evenly spaced
        vp_from_ring = np.array([self.f_pos + self.vp_sampling_radius * (np.cos(t) * second_vector + np.sin(t) * first_vector) for t in theta])
        if self.verbose_vpsampler:
            print("vp_from_ring is:", vp_from_ring)
            print("len of vp_from_ring", len(vp_from_ring))
            print("theta is:", theta)
            print("len of theta", len(theta))
        return vp_from_ring, theta

    def degree_to_radian(self, degree):
        return np.radians(degree)
    
    def filter_nonworkspace_point(self, pts, thetas):
        # Convert workspace bounds from degrees to radians
        ws_bd_a_rad = self.degree_to_radian(self.workspace_bound_a)
        ws_bd_b_rad = self.degree_to_radian(self.workspace_bound_b)
        
        # Filter the points and angles that are inside the range [workspace_bound_a, workspace_bound_b]
        filtered_vp = []
        filtered_theta = []
        for i, rad in enumerate(thetas):
            if not (ws_bd_a_rad <= rad <= ws_bd_b_rad):
                filtered_vp.append(pts[i])
                filtered_theta.append(rad)

        filtered_vps = np.array(filtered_vp)
        filtered_thetas = np.array(filtered_theta)

        if self.verbose_vpsampler:
            print("Filtered vp_from_ring:", filtered_vps)
            print("Filtered theta (radians):", filtered_thetas)
        return filtered_vps, filtered_thetas

    def set_geometric_info(self, f_pos,f_axis, obstacle_points_list, optical_pose):
        self.f_pos = f_pos
        self.f_axis = f_axis
        self.obstacle_points_list = obstacle_points_list
        self.cam_pos = optical_pose[:3]

    def set_referrence_point(self, vps, thetas):
        self.first_pt = vps[0]
        self.first_theta = thetas[0]

    def point_Circle_plane_projection(self, point, center, axis):
        return point - np.dot(point - center, axis) * axis
    
    def get_obstacle_pts_projection(self):
        obs_pts_proj = []
        for pt in self.obstacle_points_list:
            p = self.point_Circle_plane_projection(pt, self.f_pos, self.f_axis)
            obs_pts_proj.append(p)
        obs_pts_proj = np.array(obs_pts_proj)
        return obs_pts_proj
        
    def compute_line_circle_intersection(self,pt):
        dir = pt - self.f_pos               # Compute intersection points
        # Normalize point vector
        intersect_pt = self.f_pos + dir / np.linalg.norm(dir) * self.vp_sampling_radius
        return intersect_pt
    
    def get_intersection_pts_on_circle(self, obs_pts_proj_list):
        intersect_pts_list = []
        for pt in obs_pts_proj_list:
            intersect_pts_list.append(self.compute_line_circle_intersection(pt))
        intersect_pts_list = np.array(intersect_pts_list)
        return intersect_pts_list
    
    def get_point_angle(self, query_pt):
        v  = query_pt - self.f_pos
        u  = self.first_pt - self.f_pos
        dot_prod = np.dot(v, u)                 # Calculate the dot product of vectors v and u
        norm_v = np.linalg.norm(v)              # Calculate the norms (magnitudes) of each vector
        norm_u = np.linalg.norm(u)
        cos_theta = dot_prod / (norm_v * norm_u)    # Compute the cosine of the angle using the dot product
        angle_radians = np.arccos(cos_theta)        # Compute the angle in radians
        
        # if self.verbose_vpsampler:
        print("query_pt and first_pt:", query_pt, self.first_pt)
        if query_pt[1]< self.first_pt[1]:
            angle_radians += np.pi
        return angle_radians

    def get_intersection_pts_angle(self, intersect_pts):
        intersect_pts_angle = []
        for pt in intersect_pts:
            pt_rad = self.get_point_angle(pt.flatten())
            intersect_pts_angle.append(pt_rad)
        intersect_pts_angle = np.array(intersect_pts_angle)
        if self.verbose_vpsampler:
            print("intersect_pts_angle", intersect_pts_angle)
        return intersect_pts_angle
    
    def filter_vp_in_obstacle_region(self,intersect_pts_angle, vps, thetas):
        filtered_obs_vp = []        # Filter the points and angles that are inside of the obstacle range
        filtered_obs_theta = []
        for k, rad in enumerate(thetas):
            free_pts = True
            # Iterate over the obstacle_list and bring 2 elements in each iteration
            for i in range(0, len(intersect_pts_angle), 2):
                obs_min_rad = intersect_pts_angle[i]
                obs_max_rad = intersect_pts_angle[i+1]
                # print("obs_min_rad and obs_max_rad are:", obs_min_rad, obs_max_rad)
                if (obs_min_rad <= rad <= obs_max_rad):
                    free_pts = False
                    break
            if free_pts:        
                filtered_obs_vp.append(vps[k])
                filtered_obs_theta.append(thetas[k])
        filtered_obs_vp = np.array(filtered_obs_vp)
        filtered_obs_theta = np.array(filtered_obs_theta)
        return filtered_obs_vp, filtered_obs_theta

    def vp_reordering(self, vplist, thetalist):
        lower_bound = 1.5 * np.pi  # Define the boundaries: 1.5π / 2π in radians
        upper_bound = 2 * np.pi    
        
        # Find the indices for the two parts
        idxs_part1 = np.where((thetalist >= lower_bound) & (thetalist <= upper_bound))[0]
        idxs_part2 = np.where(thetalist < lower_bound)[0]
        reordered_idxs = np.concatenate((idxs_part1, idxs_part2))    # Reorder the indices
        
        # Reorder both filtered_obs_theta and filtered_obs_vp
        reordered_thetas = thetalist[reordered_idxs]
        reordered_vps = vplist[reordered_idxs]
        if self.verbose_vpsampler:
            print("reordered_vp:", reordered_vp)
        return reordered_vps,reordered_thetas

    def uniform_sampling_on_validvps(self, vplist):
        # Calculate the indices to equally sample 4 points
        idxs = [int(i * (len(vplist) - 1) / (self.num_samples - 1)) for i in range(self.num_samples)]
        print("len of vplist:", len(vplist))
        selected_vplist = np.array([vplist[i] for i in idxs])   # Get the corresponding points from the list
        return selected_vplist

    def uniform_adaptive_sampling(self):
        """
        Uniformly sampling viewpoints on the Ring
        :param f_pos: fruit position in world frame
        :param f_axis: the axis (direction) of the surrounding circle
        :param obstacle_points_list: list of obstacle points (np.array)
        :param optical_pose: the camera pose in world frame
        :return: selected_vp_list, valid_vp_list
        """
        vps_on_ring, thetas = self.get_vp_from_ring()
        self.set_referrence_point(vps_on_ring, thetas)
        filtered_vps, filtered_thetas = self.filter_nonworkspace_point(vps_on_ring, thetas)
        
        obs_pts_proj = self.get_obstacle_pts_projection()   # Conduct obstacle_points projection
        # Calculate intersection points of obstacle_points on circle
        intersect_pts = self.get_intersection_pts_on_circle(obs_pts_proj)
        # Calculate intersection point angles
        intersect_pts_angle = self.get_intersection_pts_angle(intersect_pts)
        filtered_obs_vps, filtered_obs_thetas = self.filter_vp_in_obstacle_region(intersect_pts_angle,  filtered_vps, filtered_thetas)

        reordered_vps, reordered_thetas = self.vp_reordering(filtered_obs_vps, filtered_obs_thetas)
        # Arm reachable check
        reachble_vps = self.arm_reachability_filter(reordered_vps)
        
        # Select 4 points
        selected_vp_list = self.uniform_sampling_on_validvps(reachble_vps)
        return selected_vp_list, reachble_vps
        
        
        
        
    

if __name__ == '__main__':
    tmp_mp = MyViewpointSampler(num_samples =2 )
    
    target_position = np.array([0.51, -0.26, 1.16])
    target_pose = np.array([1, 0, 0])
    target_distance = 0.35
    cam_pose = tmp_mp.predefine_start_pose(target_pose, target_distance)
    # tmp_mp.netwotk_segmentation()    
    print("cam_pose is ", cam_pose)
    print("Finish viewpoint_sampler testing...")
    # rospy.spin()
    
    # res = tmp_mp.random_sampler()
    # print("Random vp samples: ", res)
    # # Calculate the Euclidean distance between the two points
    # distance = np.linalg.norm(res[0][:3] - target_pose)
    # print("The distance between the points is:", distance)