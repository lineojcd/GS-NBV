import sys
from ultralytics import YOLO
import rospy
import cv2
import torch
import numpy as np
import ros_numpy
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, binary_dilation


class GeometryInfo:                     # Contain GeometryInfo from the current view: 
    """
    Gets data from the camera and performs semantic segmentation and pose estimation.
    """
    def __init__(self):
        model_name = rospy.get_param("Yolov8_weights_name", "myvpp3_best.pt")
        model_folder = "/home/jcd/gsnbv_ws/src/viewpoint_planning/src/perception/ultralytics_yolov8/weights/"
        model_path = model_folder +  model_name
        print("model_path is:" , model_path)
        self.model =  YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_path = "/home/jcd/gsnbv_ws/src/viewpoint_planning/src/perception/ultralytics_yolov8/images/integration/pictt.png"
        self.contour_color = torch.tensor([255, 0, 255], dtype=torch.uint8)  # Magenta color
        self.envelope_color = torch.tensor([255, 0, 0], dtype=torch.uint8)  # Blue color: BRG
        self.extract_contour = rospy.get_param("extract_contour") 
        self.contour_indicator_id = rospy.get_param("contour_indicator_id", "2")
        self.envelope_indicator_id = rospy.get_param("envelope_indicator_id", "3")
        self.fruit_peduncle_pixel_distance_threshold = rospy.get_param("fruit_peduncle_pixel_distance_threshold", "3")
        self.PIL_RGB_format = rospy.get_param("PIL_RGB_format") 
        self.cont_kernel_size = 15
        self.env_kernel_size = 7 
        self.verbose_geometric = rospy.get_param("verbose_geometric")
        self.yolo_conf_threshold = rospy.get_param("yolo_conf_threshold", "0.5")    # default: false
        self.show_yolo_redict = rospy.get_param("show_yolo_redict")    # default: false
        self.show_contour_annotated_img = rospy.get_param("show_contour_annotated_img")    # default: false
        self.contour_extract_algo = rospy.get_param("contour_extract_algo")         # default: morph_algo
        self.keypoint_dict= {}

    def get_fruit_keypoints(self):
        return self.keypoint_dict

    def plot_annotated_image(self, image, title="hi"):
        if self.PIL_RGB_format == "RGB":
            # show in RGB format in PIL
            Image.fromarray(image[..., ::-1]).show("hi")   # im_array[..., ::-1]
        if self.PIL_RGB_format == "BRG":
            # show in BRG format in PIL
            Image.fromarray(image).show(title)
    
    def draw_contour(self,annotated_image, fruit_contour,fruit_envelope):
        # Ensure the contour_color is expanded to match the dimensions of the areas being updated
        contour_color = self.contour_color.view(1, 1, 3)        # needs to be reshaped to (1, 1, 3) for broadcasting to work correctly with the mask
        envelope_color = self.envelope_color.view(1, 1, 3)

        # Use the segmentation mask contour as a boolean index mask for the annotated image
        annotated_image[fruit_contour == 2] = contour_color   # select all pixels where the contour mask is 1 and set them to the contour color
        annotated_image[fruit_envelope == 3] = envelope_color
        return annotated_image
    
    def get_fruit_contour(self,seg_masks):
        seg_masks = seg_masks[0, ...]          # Only need to get the fruit mask   | obj_nb, height, width = seg_masks.shape
        
        start_time = time.time()
        if seg_masks.is_cuda:
            seg_masks = seg_masks.cpu()               # Check if the tensor is on CUDA and move it to CPU
        np_seg_masks = seg_masks.numpy()              # Convert PyTorch tensor to NumPy array

        rospy.loginfo("Post process contour: use erosion then dilation to denoise")
        # Define the structure element for erosion and dilation
        structure_erosion = np.ones((self.cont_kernel_size, self.cont_kernel_size), dtype=bool)
        structure_dilation_c = np.ones((self.cont_kernel_size-1, self.cont_kernel_size-1), dtype=bool)
        # Apply erosion and then dilation (morphological opening)
        eroded = binary_erosion(np_seg_masks, structure=structure_erosion)
        opened_cont = binary_dilation(eroded, structure=structure_dilation_c)
        structure_dilation_e = np.ones((self.env_kernel_size, self.env_kernel_size), dtype=bool)
        opened_env = binary_dilation(opened_cont, structure=structure_dilation_e)
        # Convert back to PyTorch tensor
        tensor_cont = torch.tensor(opened_cont, dtype=torch.float32)
        tensor_env = torch.tensor(opened_env, dtype=torch.float32)
        rospy.loginfo(f"Prepare for morph_algo takes: {round((time.time() - start_time),3)} seconds")
        
        if self.contour_extract_algo == "morph_algo":
            start_time = time.time()
            cont_kernel_inner = self.cont_kernel_size - 2
            structure_dilation_cont_inner = np.ones((cont_kernel_inner, cont_kernel_inner), dtype=bool)
            opened_cont_inner = binary_dilation(eroded, structure=structure_dilation_cont_inner)
            
            env_kernel_outter = self.env_kernel_size + 2
            structure_dilation_env_outter = np.ones((env_kernel_outter, env_kernel_outter), dtype=bool)
            opened_env_outter = binary_dilation(opened_cont, structure=structure_dilation_env_outter)
            
            # Convert back to PyTorch tensor
            tensor_cont_inner = torch.tensor(opened_cont_inner, dtype=torch.float32)
            tensor_env_outter = torch.tensor(opened_env_outter, dtype=torch.float32)
            # get contour as 2 and  envelope as 3
            contour_tensor = (tensor_cont - tensor_cont_inner) * self.contour_indicator_id
            envelope_tensor = (tensor_env_outter - tensor_env) * self.envelope_indicator_id
            rospy.loginfo(f"Morph_algo for contour extraction takes: {round((time.time() - start_time),3)} seconds")
        
        rospy.logwarn("May need to remove the peduncle pixels from envelope if detected")
        return contour_tensor, envelope_tensor
    
    def get_index_from_map(self, tensor):
        ones_indices = (tensor > 0).nonzero()
        if len(ones_indices) > 0:
            top_idx = ones_indices[0]         # Find the indices of the top & bottom '1' for each column
            btm_idx = ones_indices[-1]
            
            column_indices = ones_indices[:, 1]         # Extract only the column indices
            # Find min & max column index which indicates leftmost and rightmost
            leftmost_col_idx = column_indices.min().item()
            rightmost_col_idx = column_indices.max().item() 
            # Find the corresponding row indices for the leftmost and rightmost points
            leftmost_row_idx = ones_indices[column_indices == leftmost_col_idx][0][0].item()
            rightmost_row_idx = ones_indices[column_indices == rightmost_col_idx][0][0].item()  
            return top_idx, btm_idx, leftmost_col_idx, leftmost_row_idx, rightmost_col_idx, rightmost_row_idx
        else:
            print("No '1's were found in the tensor.")
            return None, None, None, None, None, None
        
    def set_fruit_keypoints(self, contour_tensor, envelope_tensor, seg_filtered_masks):
        """
        Set Fruit keypoints:  up down left right most points of the mask
        :param contour_tensor, envelope_tensor
        :return: {'top': (310, 445), 'bottom': (515, 488), 'left': (405, 404), 'right': (410, 539), 
        #         'envelope_left': (405, 404), 'envelope_right': (410, 539)}
        """
        rospy.loginfo("Set fruit keypoints...")
        rospy.logwarn("You might need to change to 'top': (310, 445, depth_value) version")
        self.keypoint_dict= {}
        fruit_tensor = seg_filtered_masks[0, ...]
        
        # Get the index of top, bottom, left and right from the fruit: find all indices of '1's
        c_top_idx, c_btm_idx, c_lf_col_idx, c_lf_row_idx, c_rt_col_idx, c_rt_row_idx= self.get_index_from_map(contour_tensor)
        _,_, e_lf_col_idx, e_lf_row_idx, e_rt_col_idx, e_rt_row_idx= self.get_index_from_map(envelope_tensor)
        
        self.keypoint_dict['fruit_top'] = tuple(c_top_idx.tolist())
        self.keypoint_dict['fruit_bottom'] = tuple(c_btm_idx.tolist())
        self.keypoint_dict['fruit_left'] = (c_lf_row_idx, c_lf_col_idx)
        self.keypoint_dict['fruit_right'] = (c_rt_row_idx, c_rt_col_idx)
        self.keypoint_dict['fruit_contour'] = (contour_tensor > 0).nonzero()
        self.keypoint_dict['peduncle_bottom'] = None
        self.keypoint_dict['envelope_left'] = (e_lf_row_idx, e_lf_col_idx)
        self.keypoint_dict['envelope_right'] = (e_rt_row_idx, e_rt_col_idx)
        self.keypoint_dict['fruit_body'] = (fruit_tensor > 0).nonzero()
        
        if self.verbose_geometric:
            print("Keypoints info:\n",self.keypoint_dict)  
        rospy.logwarn("You may need to also pass all envelope pixels to dictionary")

    def object_class_filter(self, semantic_res, seg_masks):
        """
        Perform filtering operation for multiple class id: e.g. [(0.0, 0.969), (1.0, 0.350), (1.0, 0.421)]
        :param semantic_res, seg_masks
        :return: semantic_dict: {(class_id, class_conf)}
        :return: filtered_mask: (2,w,h)
        """
        result_list = []
        for i in range(len(semantic_res)):
            result_list.append((semantic_res[i][0], seg_masks[i,:,:]))
            
        semantic_dict = {}
        mask_dict = {}
        for idx, (key, value) in enumerate(semantic_res):
            if key not in semantic_dict:
                semantic_dict[key] = value
                mask_dict[key] = result_list[idx][1]
            else:
                if value > semantic_dict[key]:
                    semantic_dict[key] = value
                    mask_dict[key] = result_list[idx][1]
        filtered_mask = torch.stack(list(mask_dict.values()))
        return semantic_dict, filtered_mask
    
    def object_mask_filter(self, semantic_dict, seg_masks):  # Check picking score in this function
        """
        Perform picking condition check 
        :param semantic_dict, masks
        :param masks
        :return: picking_condi:  
                 segmentation_mask: (2,w,h)
        """
        print("semantic_dict is : ", semantic_dict)
        class_cnt = len(semantic_dict)
        if class_cnt == 1:
            rospy.loginfo(f"Did not detect both peduncle and fruit")
            return False, seg_masks
        else:
            avo_mask= seg_masks[0, :,:]
            peduncle_mask= seg_masks[1, :,:]
            overlap_tensor = avo_mask * peduncle_mask
            overlap_exists = overlap_tensor.nonzero().size(0) > 0   # Check if there are any non-zero elements in the result
            if overlap_exists:
                rospy.loginfo(f"Peduncle and fruit are both visiable and attaching")
                rospy.logwarn(f"TODO: max fushion(later), right now I just assign all overlap pixels to fruit")
                overlap_indices = overlap_tensor.nonzero(as_tuple=False)
                for idx in overlap_indices:
                    peduncle_mask[idx[0], idx[1]] = 0    # Assign the overlap pixels to fruit: Set 0 using the indices in peduncle_mask
                filtered_mask = torch.stack([avo_mask, peduncle_mask])
                return True, filtered_mask
            else:
                return self.check_avo_peduncle_pixel_distance(avo_mask, peduncle_mask), seg_masks
    
    def check_avo_peduncle_attaching_condition(self, pixel_distance):
        return abs(pixel_distance) < self.fruit_peduncle_pixel_distance_threshold
    
    def check_avo_peduncle_pixel_distance(self, avo_mask, peduncle_mask):
        # Find the highest part of the avo mask (the lowest index with a non-zero value)
        avo_top_row = torch.nonzero(avo_mask, as_tuple=True)[0].min().item()
        # Find the lowest part of the peduncle mask (the highest index with a non-zero value)
        peduncle_bottom_row = torch.nonzero(peduncle_mask, as_tuple=True)[0].max().item()
        # Calculate the distance between the lowest part of the peduncle and the highest part of the avo
        pixel_distance = peduncle_bottom_row - avo_top_row
        print("pixel_distance is :", pixel_distance)
        if self.check_avo_peduncle_attaching_condition(pixel_distance):
            rospy.loginfo(f"Peduncle and fruit are both visiable, the pixel distance in between is: {pixel_distance} pixels")
            return True
        else:
            rospy.logwarn(f"Peduncle and fruit are both visiable, but not attaching. You may have occlusion in between or wrong detection")
            return False
    
    def semantic_augmenation(self, fruit_contour,fruit_envelope,seg_filtered_masks,semantic_dict):
        # Augment: add label 2 as contour and assign condfi score equal to fruit
        print("show me semantic_dict: ",semantic_dict)
        try:
            semantic_dict[2.0] = semantic_dict[0.0]
            semantic_dict[3.0] = semantic_dict[0.0]
        except KeyError:
            print("Key 0.0 not found in the dictionary")
            # Optionally, set a default or handle the error appropriately
        
        # semantic_dict[2.0] = semantic_dict[0.0]
        # Augment: add label 3 as envelope and assign condfi score equal to fruit
        # semantic_dict[3.0] = semantic_dict[0.0]
        
        # Unsqueeze to make contour_mask and envelope_mask shape as (1, 3, 3)
        contour_mask = fruit_contour.unsqueeze(0)
        # contour_mask = fruit_contour.unsqueeze(0).to(seg_filtered_masks.device)
        envelope_mask = fruit_envelope.unsqueeze(0)
        
        # Concatenate all the masks along the first dimension
        seg_mask_aug = torch.cat([seg_filtered_masks, contour_mask, envelope_mask], dim=0)
        return semantic_dict,  seg_mask_aug
        
    def model_predict_ros(self,  color_image):
        """
        Perform object segmentation on the input image using YoloV8
        :param color_image: input color image: np.array
        :return: segmentation mask, list[(class_id, class_conf)]
        :return: annotated_image, 
                 segmentation_aug_mask (4,w,h)
                 semantic_aug_dict {cls_id: confi_score, }
                 picking_condi: True/False
        """
        # Perform YOLO detection (assuming your model has a method like 'predict')
        start_time = time.time()
        results = self.model(color_image,conf= self.yolo_conf_threshold)
        rospy.loginfo(f"Yolo network prediction time: {round((time.time() - start_time),3)} seconds")
        
        # Annotated_image is in RGB format
        annotated_image = results[0].plot(show=self.show_yolo_redict)        # change show=True to see annotated_image
        class_id_list =  results[0].boxes.cls.cpu().numpy()                         # tensor cpu
        class_conf_list =  results[0].boxes.conf.cpu().numpy()                      # tensor cpu
        semantic_res = list(zip(class_id_list, class_conf_list))
        print("semantic_res", semantic_res)
        if len(semantic_res) ==0:
            print("Network detected nothing. Please recheck")
            return color_image, None, None, False
        else:
            seg_masks = results[0].masks.data.cpu()                                # tensor cpu torch.Size([1, 640, 640])
            # Filtering duplicated class id, retain the id with higher conf score
            semantic_dict, seg_masks = self.object_class_filter(semantic_res, seg_masks)
            
            # Filter object mask overlapping issue and check picking condition{0,1} 
            picking_condition, seg_filtered_masks  = self.object_mask_filter(semantic_dict, seg_masks)
        
            if self.extract_contour:
                start_time = time.time()
                fruit_contour, fruit_envelope = self.get_fruit_contour(seg_filtered_masks)
                rospy.loginfo(f"Fruit contour extraction takes: {round((time.time() - start_time),3)} seconds")
                
                semantic_aug_dict,seg_masks_aug = self.semantic_augmenation(fruit_contour,fruit_envelope,seg_filtered_masks,semantic_dict)
                print("seg masks augmentation shape :",seg_masks_aug.shape, "semantic dict augmentation is:",semantic_aug_dict)
                
                # Apply the contour color to the annotated image efficiently
                annotated_image = self.draw_contour(annotated_image, fruit_contour, fruit_envelope)
                
                if self.show_contour_annotated_img:                     
                    self.plot_annotated_image(annotated_image)              # Display the result
        
            self.set_fruit_keypoints(fruit_contour, fruit_envelope, seg_filtered_masks) # Calculate fruit keypoints
        return annotated_image, seg_masks_aug, semantic_aug_dict, picking_condition





















def erode(tensor, kernel_size=3):
    padding = kernel_size // 2
    struct_elem = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)  # Full kernel
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    eroded = F.conv2d(tensor, struct_elem, padding=padding)
    return (eroded == kernel_size**2).squeeze(0).squeeze(0)  # Convert to boolean here

def dilate(tensor, kernel_size=3):
    padding = kernel_size // 2
    struct_elem = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)  # Full kernel
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    dilated = F.conv2d(tensor, struct_elem, padding=padding)
    return (dilated >= 1).squeeze(0).squeeze(0)  # Convert to boolean here



if __name__ == '__main__':
    # My test tensor
    test_tensor = torch.tensor(
    [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
     [0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]
    )
    test_tensor2 = torch.tensor(
    [[1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    )
    geoinfo =  GeometryInfo()
    contour_tensor_s = geoinfo.extract_fruit_contour_test(test_tensor)
    print( 'test_tensor')
    print( test_tensor)
    
    
    contour_tensor= torch.tensor(
       [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
        [0., 0., 0., 1., 1., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 1., 1., 0., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    print("contour_tensor_s:")
    print( contour_tensor_s)
        
    # geoinfo.get_fruit_keypoints(contour_tensor_s)    
    
    
    
    geoinfo.model_predict(True)
    # res = geoinfo.line_continuity(contour_tensor)
    # print("Continuity result if:", res)
    
    # Noise removal via erosion followed by dilation (opening)
    # Assumed noisy segmentation tensor
    noisy_segmentation_tensor = torch.tensor(
        [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
        [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
        [1., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
         dtype=torch.float32
    )
    
    # # Apply erosion to remove small noise points
    # eroded_tensor = erode(noisy_segmentation_tensor)
    # # Apply dilation to restore the shape eroded
    # cleaned_tensor = dilate(eroded_tensor)
    # print("Noisy Tensor:", cleaned_tensor)
    # print("Cleaned Tensor:", cleaned_tensor)
    