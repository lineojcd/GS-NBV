import numpy as np
from viewpoint_planners.my_method_iter import *
from utils.py_utils import look_at_rotation
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation
import rospy

class GSViewpointSampler:
    """
    Generate viewpoint samples for a given scene according to the given constraints
    """
    def __init__(self, seed: int = 2024):
        """
        Initialize the viewpoint sampler
        :param num_samples: Number of viewpoints to sample
        """
        self.num_samples = rospy.get_param("num_samples", "4")
        self.view_samples = np.empty((0, 7))
        self.seed = seed
        # np.random.seed(self.seed)
        self.pts_dict = None
        self.arm_max_reach = rospy.get_param("arm_max_reach", "0.60")
        self.arm_min_reach = rospy.get_param("arm_min_reach", "0.10")
        self.vp_sampling_radius = rospy.get_param("vp_sampling_radius", "0.20")
        self.show_plot = False 
        self.verbose_vpsampler = rospy.get_param("verbose_vpsampler")
        self.arm_origin = np.array([float(x) for x in rospy.get_param('arm_origin').split()])
        self.workspace_bound_a = rospy.get_param("workspace_bound_left", "215")
        self.workspace_bound_b = rospy.get_param("workspace_bound_right", "315")
        print("show me what is the bound", self.workspace_bound_a, self.workspace_bound_b)

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
        
        print("intersect_pts_angle", intersect_pts_angle)
        
        filtered_obs_vp = []        # Filter the points and angles that are inside of the obstacle range
        filtered_obs_theta = []
        fix_small_rad = 1.2650
        fix_big_rad = 1.4128
        for k, rad in enumerate(thetas):
            free_pts = True
            # Iterate over the obstacle_list and bring 2 elements in each iteration
            for i in range(0, len(intersect_pts_angle), 2):
                obs_min_rad = intersect_pts_angle[i]
                obs_max_rad = intersect_pts_angle[i+1]
                # print("obs_min_rad, obs_max_rad, rad", obs_min_rad, obs_max_rad, rad)
                # print("obs_min_rad and obs_max_rad are:", obs_min_rad, obs_max_rad)
                # if (obs_min_rad <= rad <= obs_max_rad):
                if (fix_small_rad <= rad <= fix_big_rad):
                    free_pts = False
                    # print("k in = ", k)
                    break
            if free_pts:        
                # print("k ok = ", k)
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

        print("len of filtered_vps and filtered_obs_vps:", len(filtered_vps), len(filtered_obs_vps))
        
        reordered_vps, reordered_thetas = self.vp_reordering(filtered_obs_vps, filtered_obs_thetas)
        # Arm reachable check
        reachble_vps = self.arm_reachability_filter(reordered_vps)
        
        # Select 4 points
        selected_vp_list = self.uniform_sampling_on_validvps(reachble_vps)
        return selected_vp_list, reachble_vps
        
        
        
        
    # TODO: delete this one later
    def vp_sampler(self,f_pos,f_axis,obstacle_points_list, optical_pose):
        """
        Generate viewpoints on the cirle
        :param f_pos: fruit position in world frame
        :param axis: the axis (direction) of the surrounding circle
        :param obstacle_points_list: list of obstacle points (np.array)
        :param optical_pose: the camera pose in world frame
        :return: None
        """
        f_pos =f_pos.flatten()   
        
        # temperary, should be changed to arm maximum reachable point coords
        tree_center =  np.array([0.5, -0.4, 0.66])
        rospy.logwarn("Use tree_center [0.5, -0.3, 0.66] for testing now")
        tree_center =  np.array([0.5, -0.3, 0.66])
        cam_posi_1 = np.array([0, 3, 0])
        
        # TODO: 1 remove tree center
        # TODO: 2 Add tree region
        
        # self.pts_dict = get_pose_info(tree_center, cam_posi_1, f_pos, obstacle_points_list, f_axis, self.vp_sampling_radius)
        
        self.pts_dict = get_pose_info(tree_center, cam_posi_1, f_pos, obstacle_points_list, f_axis, self.vp_sampling_radius)
        vp_pts = self.pts_dict["vp_pts"]
        obs_in_range = self.pts_dict["obstacle_in_range"]
        ws_in_range = self.pts_dict["workspace_in_range"]
        ws_inters_a = self.pts_dict["workspace_bound_pta"]
        ws_inters_b = self.pts_dict["workspace_bound_ptb"]
        obs_inters = self.pts_dict["obstacle_pts_intersect_on_circle"]
        valid_viewpoint_list = vp_pts[~obs_in_range & ~ws_in_range]
        
        # validVP_x, validVP_y, validVP_z  = vp_pts[~obs_in_range & ~ws_in_range, 0], vp_pts[~obs_in_range & ~ws_in_range, 1], vp_pts[~obs_in_range & ~ws_in_range, 2]
        # print(">>>  vp_pts[~obs_in_range & ~ws_in_range]",  )
        
        filtered_viewpoint_list = self.arm_reachability_filter(valid_viewpoint_list)
        
        # vp_list = [ws_inters_a,obs_inters[0], obs_inters[1], ws_inters_b ]
        vp_list = [filtered_viewpoint_list[0],obs_inters[0], obs_inters[1], filtered_viewpoint_list[-1] ]
        
        my_vp_pose_list = []
        for posi in vp_list:
            my_orien = my_look_at_rotation(posi, f_pos)
            if self.verbose_vpsampler:
                print(" my_nbv_sampler my_look_at_rotation: ",my_orien)
            my_pose = np.concatenate((posi, my_orien))
            my_vp_pose_list.append(my_pose)
        
        valid_pose_list = []
        for posi in filtered_viewpoint_list:
            orien = my_look_at_rotation(posi, f_pos)
            pose = np.concatenate((posi, orien))
            valid_pose_list.append(pose)
        return my_vp_pose_list, valid_pose_list
    

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