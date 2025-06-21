import sys
from ultralytics import YOLO
import rospy
import cv2
import torch
import numpy as np
import ros_numpy
import time
import os
import math
from sensor_msgs.msg import Image
from scene_representation.conversions import T_from_rot_trans_np
from perception.my_geometry_info import GeometryInfo
from perception.my_bounded_axis_test import get_bounded_fruit_axis
from perception.my_realsense_ros_capturer import MyRealsenseROSCapturer
from cv_bridge import CvBridge, CvBridgeError
from utils.torch_utils import transform_from_rotation_translation, transform_from_rot_trans, generate_pixel_coords, get_centroid, transform_points, transform_points_cam_to_world_gnbv, transform_points_cam_to_world_my
from scipy.spatial.transform import Rotation as R
from viewpoint_planners.fit_3dline import fit_3dline_by_PCA, fit_3dline_by_2points
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class MyPerceiver:
    """
    Gets data from the camera and performs semantic segmentation and pose estimation.
    """
    def __init__(self):
        self.capturer = MyRealsenseROSCapturer()
        # self.device = "cpu"            #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arm_type = rospy.get_param('arm_type')
        self.camera_type = rospy.get_param('camera_type')
        self.bridge = CvBridge()
        self.max_depth_value = rospy.get_param("max_depth_value", "1")   #Here 1 is an integer
        self.filter_depth = rospy.get_param("filter_depth")
        self.smooth_depth = rospy.get_param("smooth_depth")
        self.filter_fruit_noise = rospy.get_param("filter_fruit_noise")
        self.filter_noise_level = rospy.get_param("filter_noise_level", "1")
        self.occlusion_depth_offset = rospy.get_param("occlusion_depth_offset", "0.04")
        self.geo_info = GeometryInfo()
        self.shift_pixel = rospy.get_param("shift_optical_pixel")
        self.axis_estimation = rospy.get_param("axis_estimation_method")
        self.fruit_position_estimation = rospy.get_param("fruit_position_estimation_method")
        self.verbose_perceiver = rospy.get_param("verbose_perceiver")
        self.cls_bg_label = rospy.get_param("classification_background_label", "-1")   
        self.cls_conf_lodds = rospy.get_param("classification_background_label_confidence_logodds", "0.0")   #default prob 50%
        
    def log_odds(self, p):
        return np.log(p / (1 - p))      # log_odds(0.4): -0.4; log_odds(0.9): 2.2; log_odds(0.2) = -1.4

    def get_camera_info(self):
        camera_info, _, _ = self.capturer.get_frames()
        self.width = camera_info.width
        self.height = camera_info.height
        self.intrinsics = torch.tensor(np.array(camera_info.K).reshape(3, 3),dtype=torch.float32,device=torch.device("cuda:0"))
        return camera_info

    def filter_depth_image(self, depth_image):
        # Realsense L515: Use np.where to replace NaN values by max_depth_value
        if self.camera_type == 'l515':
            return np.where(np.isnan(depth_image), self.max_depth_value, depth_image)
        
        # Realsense D435: Replace 9999 with max_depth_value
        if self.camera_type == 'd435':
            return np.where(depth_image > self.max_depth_value, self.max_depth_value, depth_image)
            # return np.where((depth_image > self.max_depth_value) | (depth_image == 0),self.max_depth_value, depth_image)

    def publish_annotated_img(self,annotated_image):
        try:
            # Convert the processed CV2 image back to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")  
            self.capturer.image_pub.publish(ros_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def get_sensor_data(self):
        camera_info, color_output, depth_output = self.capturer.get_frames()
        color_image = color_output["color_image"]
        depth_image = depth_output["depth_image"]                   # np.array
        points = depth_output["points"]
        
        # Return if no data
        if camera_info is None or color_image is None:
            rospy.logwarn("[Perceiver] Perception paused. No data from camera.")
            return None, None, None, None
        return camera_info, color_image, depth_image, points

    def show_depth_image_stats(self, depth_img):
        print(f"Mean: {np.mean(depth_img):.3f}, Min: {np.min(depth_img)}, Max: {np.max(depth_img)}, Std: {np.std(depth_img):.3f}")

    def depth_image_preprocess(self,depth_image):
        if self.verbose_perceiver:
            print("depth image before filter stats:")
            self.show_depth_image_stats(depth_image)
        
        if self.filter_depth:
            preprocessed_depth_img = self.filter_depth_image(depth_image)    
            print("depth image after filter stats:")
            self.show_depth_image_stats(preprocessed_depth_img)
            
        if self.smooth_depth:
            preprocessed_depth_img = gaussian_filter(preprocessed_depth_img, sigma=1)
        return preprocessed_depth_img
        
    def run(self):
        camera_info, color_image, depth_image, points = self.get_sensor_data()          # Get data from camera, np.array
        
        if camera_info is None or color_image is None or depth_image is None:
            return None, None, None, None, None
        preprocessed_depth_img = self.depth_image_preprocess(depth_image)
        
        # Yolo segmentation
        annotated_img, seg_masks, semantic_res_dict,picking_condi = self.geo_info.model_predict_ros(color_image)
        rospy.loginfo(f"picking_condi is : {picking_condi} ")
        self.publish_annotated_img(annotated_img)
        
        keypoint_dict = self.geo_info.get_fruit_keypoints()
        start_time = time.time()
        occlusion_rate, occluded_pts = self.get_occlusion_rate_and_points(preprocessed_depth_img, seg_masks)
        rospy.loginfo(f"valid occlusion rate is: {occlusion_rate},  compute occlusion rate takes: {round((time.time() - start_time),4)} seconds")
        picking_score = picking_condi * (1 - occlusion_rate)
        rospy.loginfo(f"Picking score in current view is : {picking_score} by picking_condi * (1 - occlusion_rate)")
        
        keypoint_dict = {**keypoint_dict, **occluded_pts}
        if self.verbose_perceiver:
            print("keypoint_dict is:", keypoint_dict)
        
        start_time = time.time()
        semantics = self.assign_semantics(camera_info, seg_masks,semantic_res_dict)  
        rospy.loginfo(f"Assign semantics for network detector takes: {round((time.time() - start_time),4)} seconds")
        return preprocessed_depth_img,  keypoint_dict,semantics, (picking_condi, occlusion_rate)

    def get_contour_envelope_depth_stat_info(self, contour_depth_v_set, envelope_depth_v_set):
        if self.filter_depth:
            # Remove the element with max_depth_value using discard()
            contour_depth_v_set.discard(self.max_depth_value)
            fruit_min_depth = round(min(contour_depth_v_set),4)
            fruit_max_depth = round(max(contour_depth_v_set),4)
            enve_min_depth = round(min(envelope_depth_v_set),4)
            enve_max_depth = round(max(envelope_depth_v_set),4)
        else:
            # Remove NaN values using set comprehension and math.isnan
            contour_depth_v_set_filtered = {x for x in contour_depth_v_set if not math.isnan(x)}
            envelope_depth_v_set_filtered = {x for x in envelope_depth_v_set if not math.isnan(x)}
            fruit_min_depth = round(min(contour_depth_v_set_filtered),4)
            fruit_max_depth = round(max(contour_depth_v_set_filtered),4)
            enve_min_depth = round(min(envelope_depth_v_set_filtered),4)
            enve_max_depth = round(max(envelope_depth_v_set_filtered),4)
        return fruit_min_depth, fruit_max_depth, enve_min_depth, enve_max_depth

    def get_occlusion_rate_and_points(self,depth_image, seg_masks):
        if seg_masks is None:
            rospy.logerr("No object")
            return -1, {'obstacle_points_left': None, 'obstacle_points_right': None}
        layer = seg_masks.shape[0]
        if layer <3 :
            rospy.logerr("No occlusion")
            return -1, {'obstacle_points_left': None, 'obstacle_points_right': None}
        
        valid_cont_idxes, valid_cont_depth_values, valid_env_idxes, valid_env_depth_values = self.get_valid_contour_envelope_indexes_and_depth(depth_image, seg_masks)
        
        valid_cont_depth_values_set = set(valid_cont_depth_values.flatten())
        valid_env_depth_values_set = set(valid_env_depth_values.flatten())
        valid_cont_minD, valid_cont_maxD, valid_env_minD, valid_env_maxD = self.get_contour_envelope_depth_stat_info(valid_cont_depth_values_set,valid_env_depth_values_set)
        print("After filter: fruit contour min & max depth are:",valid_cont_minD ," and ", valid_cont_maxD)
        print("After filter: fruit envelope min & max depth are:",valid_env_minD ," and ", valid_env_maxD)
        print(f"Total valid points for contour & envelope are: {valid_cont_depth_values.size}  and {valid_env_depth_values.size}")
        
        occ_rate, occluded_pts_indexes = self.compute_occlusion_rate(valid_env_depth_values, valid_env_idxes, valid_cont_maxD, valid_cont_minD)
        print("occ_rate and occluded_pts_indexes are: ", occ_rate, occluded_pts_indexes)
        occluded_pts_dict = self.set_occluded_points(occluded_pts_indexes)
        print("occluded_pts_dict are: ", occluded_pts_dict)
        return round(occ_rate,4), occluded_pts_dict

    def get_valid_contour_envelope_indexes_and_depth(self,depth_image, seg_masks):
        contour_mask = seg_masks[-2]
        envelope_mask = seg_masks[-1] 
        cont_idxes = contour_mask.nonzero(as_tuple=False)
        env_idxes = envelope_mask.nonzero(as_tuple=False)
        cont_depth_values = depth_image[cont_idxes[:, 0], cont_idxes[:, 1]]
        env_depth_values = depth_image[env_idxes[:, 0], env_idxes[:, 1]]
        
        # Apply a mask to filter out 0 and 1 values
        valid_cont_depth_mask = (cont_depth_values != 0) & (cont_depth_values != 1)
        valid_env_depth_mask = (env_depth_values != 0) 
        
        # Get the indices and values where the depth values are valid
        valid_cont_idxes = cont_idxes[valid_cont_depth_mask]
        valid_cont_depth_values = cont_depth_values[valid_cont_depth_mask]
        valid_env_idxes = env_idxes[valid_env_depth_mask]
        valid_env_depth_values = env_depth_values[valid_env_depth_mask]
        
        if self.verbose_perceiver:
            cont_depth_values_set = set(cont_depth_values.flatten())
            env_depth_values_set = set(env_depth_values.flatten())
            cont_minD, cont_maxD, env_minD, env_maxD = self.get_contour_envelope_min_max_depth(cont_depth_values_set,env_depth_values_set)
            print("before filter")
            print("fruit contour min and max depth are:",cont_minD ," and ", cont_maxD)
            print("fruit envelope min and max depth are:",env_minD ," and ", env_maxD)
            print(f"Total points for contour and envelope are: {cont_depth_values.size}  and {env_depth_values.size}")
            
        return valid_cont_idxes, valid_cont_depth_values, valid_env_idxes, valid_env_depth_values
    
    def compute_occlusion_rate(self, valid_env_depth_values, valid_env_idxes, valid_cont_maxD, valid_cont_minD):
        occluded_pts_indexes = []       # List to store pixel indexes where the condition is met
        free_pt_cnt = 0
        rospy.logwarn("I changed the valid_cont_maxD in env_pt.item() > valid_cont_maxD + self.occlusion_depth_offset to valid_cont_minD")
        for i, env_pt in enumerate(valid_env_depth_values):
            # print(env_pt.item(), "><", valid_cont_maxD , "+", self.occlusion_depth_offset, "=", valid_cont_maxD + self.occlusion_depth_offset)
            if env_pt.item() > valid_cont_minD + self.occlusion_depth_offset:
                free_pt_cnt += 1
            # if env_pt.item() < valid_cont_minD:
            else:
                occluded_pts_indexes.append(valid_env_idxes[i].tolist())
        occ_rate = 1 - free_pt_cnt/valid_env_depth_values.size
        return occ_rate, occluded_pts_indexes
    
    def set_occluded_points(self, occluded_pts_indexes): # Keep leftmost and rightmost occluded points
        leftmost_index = None
        rightmost_index = None
        if occluded_pts_indexes:
            occluded_pts_indexes.sort(key=lambda idx: idx[1])  # Sort the pts based on the x-coord (column index)
            leftmost_index = occluded_pts_indexes[0]            # Store the left & right pts index into dictionary
            rightmost_index = occluded_pts_indexes[-1]
            occluded_pts = {'obstacle_points_left': leftmost_index, 'obstacle_points_right': rightmost_index}
        else:
            occluded_pts = {'obstacle_points_left': None, 'obstacle_points_right': None}  # If no occluded pts indexes, set None
        return occluded_pts
    
    def assign_semantics(self, camera_info, seg_masks,semantic_res_dict) -> torch.tensor:
        """
        Assign the confidence scores and labels to the pixels in the image
        :param camera_info: camera information: height and width
        :param seg_masks: segmentation mask [H x W]   / [No x H x W]
        :param semantic_res: list[(class_id, class_conf)] 
        :return: semantic confidence scores and labels [H x W x 2]
        """
        image_size = (camera_info.height, camera_info.width)
        
        # Create a mask that is log odds 0.9 if there's a semantic value and log odds of 0.4 (free) otherwise
        
        # Init: default class ci = −1 (background) and Ps(i) = 0.5 were assigned
        label_mask = self.cls_bg_label * torch.ones(image_size, dtype=torch.float32)
        # label_mask = self.cls_bg_label * torch.ones(image_size, dtype=torch.float32, device=self.device)
        conf_lodds_mask = self.cls_conf_lodds * torch.ones(image_size, dtype=torch.float32)
        # conf_lodds_mask = self.cls_conf_lodds * torch.ones(image_size, dtype=torch.float32, device=self.device)
        
        if  (semantic_res_dict is None) or (len(semantic_res_dict) == 0)  :
            rospy.logerr("No OOI detected in current view, only backgrounds")
            return None
        else:
            semantic_res = list(semantic_res_dict.items())
            for obj in range(len(semantic_res)):
                # Assign the semantic class_conf and class_id: # 0:fruit; 1:peduncle; -1:bg; 2:cont; 3: env
                class_id = semantic_res[obj][0]     # Extract the class_id and the log odds of the class_conf
                class_conf_odds = self.log_odds(semantic_res[obj][1])
                print("cls id is :", class_id)
                
                # Create a mask for the current object where the segmntation mask is greater than 0
                current_mask = seg_masks[obj] > 0
                
                # Max fusion: Assign overlapping pixels to the object with higher conf
                # Check where the new confidence is greater than what's already in the conf_score_odds_mask
                update_mask = (current_mask & (class_conf_odds >= conf_lodds_mask))
                
                label_mask[update_mask] = torch.tensor(class_id, device=conf_lodds_mask.device)
                conf_lodds_mask[update_mask] = class_conf_odds
                
            # At the same spot (env, 0.96) , (peduncle, 0.71), but you need to use peduncle instead of env
            rospy.logwarn("You may need to handle the envelope and peduncle issue later")
            semantics = torch.stack((conf_lodds_mask, label_mask), dim=-1)
            return semantics

    def estimate_fruit_position(self, pixel_coords, cam_transform_in_world,  depth_img, dict_value, batch_size, verbose=False):
        # Separate the indices into row and column indices
        rows = dict_value[:, 0]
        cols = dict_value[:, 1]
        
        # Use the indices to extract elements from pixel_coords
        fruit_pixel_coords = pixel_coords[rows, cols].view(-1, 3) @ torch.inverse(self.intrinsics).t().type(torch.float32)
        fruit_depth = depth_img[rows, cols].view(1, -1)
        
        nb_fruit_depth_pts = fruit_depth.numel()
        # print("nb_fruit_depth_pts:", nb_fruit_depth_pts)
        # print("fruit_pixel_coords:", fruit_pixel_coords)
        
        # Filter the fruit_depth
        if torch.any((fruit_depth == self.max_depth_value) | (fruit_depth == 0)):
            rospy.loginfo("Filtering invlid fruit depth value (0 or max depth value) before estimate fruit position... ")
            # Create a boolean mask to filter out the values equal to self.max_depth_value (3)
            depth_filter_mask = (fruit_depth != self.max_depth_value) & (fruit_depth != 0)
            fruit_depth = fruit_depth[depth_filter_mask]
            nb_filtered_depth_pts = nb_fruit_depth_pts - fruit_depth.numel()
            print("Total pts:",nb_fruit_depth_pts, "filtered pts:", nb_filtered_depth_pts, "remaining pts:", fruit_depth.numel())
            fruit_pixel_coords = fruit_pixel_coords[depth_filter_mask.flatten()]
        
        fruit_cam_coords = (fruit_pixel_coords * fruit_depth.unsqueeze(-1)).view(batch_size, -1, 3)
        if verbose:
            print("fruit_cam_coords:\n",fruit_cam_coords)
        fruit_world_coords = transform_points_cam_to_world_my(fruit_cam_coords, cam_transform_in_world, self.arm_type).cpu()
        return fruit_world_coords
            
    def estimate_fruit_pose(self,depth_img, keypt_dict, cam_pose, device = torch.device("cuda:0")):
        """
        Get fruit pose
        """
        position = torch.tensor(cam_pose[:3], dtype=torch.float32, device=device)
        orientation = torch.tensor(R.from_quat(cam_pose[3:]).as_matrix(), dtype=torch.float32, device=device)
        # get camera pose in 4*4 matrice
        Trans_w_o = transform_from_rot_trans(orientation[None, :], position[None, :])
        depth_img = torch.tensor(depth_img, dtype=torch.float32, device=device)
        
        batch_size = 1
        pixel_coords = generate_pixel_coords(self.width, self.height, device, self.shift_pixel)  # ([w640, h640, 3])
        
        fruit_idx = keypt_dict[self.fruit_position_estimation]              # fruit_body or fruit_contour
        fruit_world_coords = self.estimate_fruit_position(pixel_coords, Trans_w_o, depth_img, fruit_idx, batch_size)
        # print("fruit_world_coords:\n",fruit_world_coords)
        f_pos = get_centroid(fruit_world_coords, self.filter_fruit_noise, self.filter_noise_level).numpy()
        
        bounded_axis = self.estimate_fruit_axis(fruit_world_coords)         # Get bounded fruit axis  
        return f_pos, bounded_axis
    
    def estimate_fruit_axis(self, fruit_points_in_wf):
        points = fruit_points_in_wf[0]                  # Extract the points: Remove the batch dimension
        z_values = points[:, 2]                         # Extract z-values (the third column)
        max_z_index = torch.argmax(z_values).item()     # Find the index of the highest z-value
        min_z_index = torch.argmin(z_values).item()     # Find the index of the lowest z-value
        top_pt = points[max_z_index]                    # Get the points with the highest and lowest z-values
        bottom_pt = points[min_z_index]
        
        print("top_pt and bottom_pt are :", top_pt, bottom_pt)
        if self.axis_estimation == "PCA":
            axis = fit_3dline_by_PCA(fruit_points_in_wf)
        if self.axis_estimation == "2points":
            axis = fit_3dline_by_2points(top_pt, bottom_pt)
        bounded_axis = get_bounded_fruit_axis(axis, plot = False)
        return bounded_axis
    
    def estimate_obstacle_points(self,depth_img, keypt_dict, cam_pose, device = torch.device("cuda:0")):
        position = torch.tensor(cam_pose[:3], dtype=torch.float32, device=device)
        orientation = torch.tensor(R.from_quat(cam_pose[3:]).as_matrix(), dtype=torch.float32, device=device)
        # get camera pose in 4*4 matrice
        Trans_w_o = transform_from_rot_trans(orientation[None, :], position[None, :])
        depth_img = torch.tensor(depth_img, dtype=torch.float32, device=device)
        batch_size = 1
        pixel_coords = generate_pixel_coords(self.width, self.height, device, self.shift_pixel)      # ([640, 640, 3])
        
        if self.verbose_perceiver:
            keys_list = list(keypt_dict.keys())
            print("keypt_dict keys are ", keys_list)
        
        lf_idx = keypt_dict['obstacle_points_left']     # Separate the indices into row and column indices
        rt_idx = keypt_dict['obstacle_points_right']
        
        if lf_idx is not None and rt_idx is not None:
            # Use the indices to extract elements from pixel_coords
            lf_obs_cam_coords = pixel_coords[lf_idx[0], lf_idx[1]].view(-1, 3) @ torch.inverse(self.intrinsics).t().type(torch.float32)
            lf_obs_depth = depth_img[lf_idx[0], lf_idx[1]].view(1, -1)
            lf_obs_cam = (lf_obs_cam_coords * lf_obs_depth.unsqueeze(-1)).view(batch_size, -1, 3)
            lf_obs_wf = transform_points_cam_to_world_my(lf_obs_cam, Trans_w_o, self.arm_type).cpu()
            rt_obs_cam_coords = pixel_coords[rt_idx[0], rt_idx[1]].view(-1, 3) @ torch.inverse(self.intrinsics).t().type(torch.float32)
            rt_obs_depth = depth_img[rt_idx[0], rt_idx[1]].view(1, -1)
            rt_obs_cam = (rt_obs_cam_coords * rt_obs_depth.unsqueeze(-1)).view(batch_size, -1, 3)
            rt_obs_wf = transform_points_cam_to_world_my(rt_obs_cam, Trans_w_o, self.arm_type).cpu()
        
            # Extracting the first row from each tensor and converting to numpy arrays
            lf_obs_np = lf_obs_wf[0].numpy()      # Result: array([[ 0.4894, -0.1412, 1.1822]])
            rt_obs_np = rt_obs_wf[0].numpy()    # Result: array([[ 0.4834, -0.1412, 1.1739]])
            # obs_pts_array = np.vstack((lf_obs_np, rt_obs_np))       # Combine them into a single numpy array
            obs_pts_array = [lf_obs_np, rt_obs_np]                  # Combine them into a list
        else: 
            obs_pts_array = None
        return obs_pts_array
    
    













if __name__ == '__main__':
    rospy.init_node('realsense_yolo_processor', anonymous=True)
    tmp_mp = MyPerceiver()
    tmp_mp.run()
    # tmp_mp.netwotk_segmentation()    
    print("Finish my_perceiver.py testing...")
    # rospy.spin()