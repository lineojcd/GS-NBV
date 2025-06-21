import numpy as np
from viewpoint_planners.my_method_iter import *
from utils.py_utils import look_at_rotation
from viewpoint_planners.my_look_at_rotation import my_look_at_rotation
import rospy

class MyViewpointSampler:
    """
    Generate viewpoint samples for a given scene according to the given constraints
    """
    def __init__(self, num_samples: int = 1, seed: int = 2024):
        """
        Initialize the viewpoint sampler
        :param num_samples: Number of viewpoints to sample
        """
        # TODO: can set this one later
        self.num_samples = rospy.get_param("num_samples", "4")
        self.view_samples = np.empty((0, 7))
        self.seed = seed
        # np.random.seed(self.seed)
        self.pts_dict = None
        self.arm_max_reach = rospy.get_param("arm_max_reach", "0.60")
        self.vp_sampling_radius = rospy.get_param("vp_sampling_radius", "0.21")
        self.verbose_vpsampler = rospy.get_param("verbose_vpsampler")
        self.arm_origin = np.array([float(x) for x in rospy.get_param('arm_origin').split()])  

    def arm_reachability_filter(self, valid_viewpoint_list):
        # Filter the valid_viewpoint_list by arm reachability
        return [vp for vp in valid_viewpoint_list if np.linalg.norm(vp - self.arm_origin) <= self.arm_max_reach]


    def vp_sampler(self,f_pos,f_axis: np.array = np.array([0.5, -0.4, 1.0]),
        obstacle_points: np.array = np.array([[1.5, 1.5, 0], [-1.5, 1.5, 0]])) -> np.ndarray:
        """
        Generate viewpoints on the cirle
        :param target_posi: Target position in world frame
        :param axis: the axis (direction) of the surrounding circle
        :param obstacle_points: the obstacle points 
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
        
        self.pts_dict = get_pose_info(tree_center, cam_posi_1, f_pos, obstacle_points, f_axis, self.vp_sampling_radius)
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