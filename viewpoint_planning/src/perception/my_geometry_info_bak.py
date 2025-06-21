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

# Contain GeometryInfo from the current view: 
class GeometryInfo:
    """
    Gets data from the camera and performs semantic segmentation and pose estimation.
    """

    def __init__(self):
        model_path = "/home/jcd/gradient_nbv_ws/src/gradientnbv/src/viewpoint_planning/src/perception/ultralytics_yolov8/weights/myvpp_sim_best.pt"
        self.model =  YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.image_path = "/home/jcd/gradient_nbv_ws/src/gradientnbv/src/viewpoint_planning/src/perception/ultralytics_yolov8/images/integration/test_p1.png"
        self.image_path = "/home/jcd/gradient_nbv_ws/src/gradientnbv/src/viewpoint_planning/src/perception/ultralytics_yolov8/images/integration/pictt.png"
        self.contour_color = torch.tensor([255, 0, 255], dtype=torch.uint8)  # Magenta color
        self.envelope_color = torch.tensor([255, 0, 0], dtype=torch.uint8)  # Blue color: BRG
        self.extract_contour = rospy.get_param("extract_contour") 
        self.plot_RGB = True
        self.contour_post_process = True
        self.cont_kernel_size = 15
        self.env_kernel_size = 7 
        self.verbose_perceiver = rospy.get_param("verbose_perceiver")
        self.yolo_conf_threshold = rospy.get_param("yolo_conf_threshold", "0.5")    # default: false
        self.show_yolo_redict = rospy.get_param("show_yolo_redict")    # default: false
        self.show_contour_annotated_img = rospy.get_param("show_contour_annotated_img")    # default: false
        self.contour_extract_algo = rospy.get_param("contour_extract_algo")         # default: morph_algo
        self.neighbors = [  (-1, -1),   (-1, 0),    (-1, 1),            # Directions for the 8 neighbors
                            (0, -1),                (0, 1),
                            (1, -1),    (1, 0),     (1, 1)]
        self.keypoint_dict= {}
        self.verbose = False
        self.line_3d = None

    def get_fruit_keypoints(self):
        return self.keypoint_dict

    def plot_annotated_image(self, image, title="hi"):
        if self.plot_RGB :
            # show in RGB format in PIL
            Image.fromarray(image[..., ::-1]).show("hi")   # im_array[..., ::-1]
        else:
            # show in BRG format in PIL
            Image.fromarray(image).show(title)
    
    def draw_contour(self,annotated_image, fruit_contour,fruit_envelope):
        # Ensure the contour_color is expanded to match the dimensions of the areas being updated
        # contour_color needs to be reshaped to (1, 1, 3) for broadcasting to work correctly with the mask
        contour_color = self.contour_color.view(1, 1, 3)
        envelope_color = self.envelope_color.view(1, 1, 3)

        # Use the segmentation mask contour as a boolean index mask for the annotated image
        # This will select all pixels where the contour mask is 1 and set them to the contour color
        annotated_image[fruit_contour == 2] = contour_color
        annotated_image[fruit_envelope == 3] = envelope_color
        return annotated_image
    
    def get_fruit_contour(self,tensor):
        obj_nb, height, width = tensor.shape
        # Only need to get the fruit mask
        tensor = tensor[0, ...]
        
        if self.contour_post_process:
            start_time = time.time()
            # Check if the tensor is on CUDA and move it to CPU
            if tensor.is_cuda:
                tensor = tensor.cpu()
                
            # Convert PyTorch tensor to NumPy array
            np_tensor = tensor.numpy()

            rospy.loginfo("Post process contour: use erosion then dilation to denoise")
            # Define the structure element for erosion and dilation
            structure_erosion = np.ones((self.cont_kernel_size, self.cont_kernel_size), dtype=bool)
            structure_dilation_c = np.ones((self.cont_kernel_size-1, self.cont_kernel_size-1), dtype=bool)
            # Apply erosion and then dilation (morphological opening)
            eroded = binary_erosion(np_tensor, structure=structure_erosion)
            opened_cont = binary_dilation(eroded, structure=structure_dilation_c)
            
            structure_dilation_e = np.ones((self.env_kernel_size, self.env_kernel_size), dtype=bool)
            opened_env = binary_dilation(opened_cont, structure=structure_dilation_e)

            # Convert back to PyTorch tensor
            tensor_cont = torch.tensor(opened_cont, dtype=torch.float32)
            tensor_env = torch.tensor(opened_env, dtype=torch.float32)
        
            print(f"morph prepare time: {time.time() - start_time} seconds")
        
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
            
            # get contour and  envelope
            contour_tensor = (tensor_cont - tensor_cont_inner) * 2
            envelope_tensor = (tensor_env_outter - tensor_env) * 3

            elapsed_time = time.time() - start_time
            print(f"morph_algo for contour and envelope time: {elapsed_time} seconds")
            
        else:
            # "neighbor_algo"
            # Iterate over each pixel in the tensor to get contour
            start_time = time.time()
            contour_tensor = torch.zeros((height, width), dtype=torch.float32)
            envelope_tensor = torch.zeros((height, width), dtype=torch.float32)
            for i in range(height):
                for j in range(width):
                    # Check only pixels that are '1'
                    if tensor_cont[i, j] == 1:
                        # Initialize a flag to check boundary conditions
                        is_contour = False
                        
                        # Check if the current pixel is at the boundary of the tensor
                        if i == 0 or i == height-1 or j == 0 or j == width-1:
                            is_contour = True  # Consider boundary pixels as part of the contour
                        else:
                            # Check the neighbors (up, down, left, right)
                            if tensor_cont[i-1, j] == 0 or tensor_cont[i+1, j] == 0 or tensor_cont[i, j-1] == 0 or tensor_cont[i, j+1] == 0:
                                is_contour = True
                        
                        # If it's a contour (boundary or has zero neighbors), set it
                        if is_contour:
                            contour_tensor[i, j] = 2
            
            # Iterate over each pixel in the tensor to get envelope
            for i in range(height):
                for j in range(width):
                    # Check only pixels that are '1'
                    if tensor_env[i, j] == 1:
                        # Initialize a flag to check boundary conditions
                        is_envelope= False
                        
                        # Check if the current pixel is at the boundary of the tensor
                        if i == 0 or i == height-1 or j == 0 or j == width-1:
                            is_envelope = True  # Consider boundary pixels as part of the contour
                        else:
                            # Check the neighbors (up, down, left, right)
                            if tensor_env[i-1, j] == 0 or tensor_env[i+1, j] == 0 or tensor_env[i, j-1] == 0 or tensor_env[i, j+1] == 0:
                                is_envelope = True
                        
                        # If it's a contour (boundary or has zero neighbors), set it
                        if is_envelope:
                            envelope_tensor[i, j] = 3
                            
            elapsed_time = time.time() - start_time
            print(f"neighbor_algo time: {elapsed_time} seconds")

        print("TODO:May need to remove the peduncle pixels from envelope if detected")
        return contour_tensor, envelope_tensor
    
    def get_index_from_map(self, tensor):
        ones_indices = (tensor > 0).nonzero()
        if len(ones_indices) > 0:
            # Find the indices of the top and bottom '1' for each column
            top_idx = ones_indices[0]
            btm_idx = ones_indices[-1]
            
            # Extract only the column indices
            column_indices = ones_indices[:, 1]
            # Find minimum and maximum column index which indicates leftmost and rightmost
            leftmost_col_idx = column_indices.min().item()
            rightmost_col_idx = column_indices.max().item() 
            # Find the corresponding row indices for the leftmost and rightmost points
            leftmost_row_idx = ones_indices[column_indices == leftmost_col_idx][0][0].item()
            rightmost_row_idx = ones_indices[column_indices == rightmost_col_idx][0][0].item()  
        else:
            print("No '1's were found in the tensor.")
        return top_idx, btm_idx, leftmost_col_idx, leftmost_row_idx, rightmost_col_idx, rightmost_row_idx
        
    #TODO: May need to change to 'top': (310, 445, depth_value) version
    def set_fruit_keypoints(self, contour_tensor, envelope_tensor, seg_filtered_masks):
        """
        Set Fruit keypoints:  up down left right most points of the mask
        :param contour_tensor, envelope_tensor
        :return: {'top': (310, 445), 'bottom': (515, 488), 
        #         'left': (405, 404), 'right': (410, 539), 
        #         'envelope_left': (405, 404), 'envelope_right': (410, 539)}
        """
        print("Set fruit keypoints...")
        self.keypoint_dict= {}
        fruit_tensor = seg_filtered_masks[0, ...]
        # # Find the minimum value in the tensor
        # min_value = torch.min(fruit_tensor)
        # # Find the maximum value in the tensor
        # max_value = torch.max(fruit_tensor)
        # ones_indices_f = (fruit_tensor > 0).nonzero()
        
        # print("fruit_tensor shape: ", fruit_tensor.shape)
        # print("Min max of this fruit_tensor", min_value, max_value )
        # print("contour_tensor shape: ", contour_tensor.shape)
        # print("envelope_tensor shape: ", envelope_tensor.shape)
        
        # Get the index of top, bottom, left and right from the fruit
        # Find all indices of '1's
        c_top_idx, c_btm_idx, c_lf_col_idx, c_lf_row_idx, c_rt_col_idx, c_rt_row_idx= self.get_index_from_map(contour_tensor)
        _, _, e_lf_col_idx, e_lf_row_idx, e_rt_col_idx, e_rt_row_idx= self.get_index_from_map(envelope_tensor)
        # _, _, f_lf_col_idx, f_lf_row_idx, f_rt_col_idx, f_rt_row_idx= self.get_index_from_map(fruit_tensor)
        
        self.keypoint_dict['fruit_top'] = tuple(c_top_idx.tolist())
        self.keypoint_dict['fruit_bottom'] = tuple(c_btm_idx.tolist())
        self.keypoint_dict['fruit_left'] = (c_lf_row_idx, c_lf_col_idx)
        self.keypoint_dict['fruit_right'] = (c_rt_row_idx, c_rt_col_idx)
        self.keypoint_dict['fruit_contour'] = (contour_tensor > 0).nonzero()
        self.keypoint_dict['peduncle_bottom'] = None
        self.keypoint_dict['envelope_left'] = (e_lf_row_idx, e_lf_col_idx)
        self.keypoint_dict['envelope_right'] = (e_rt_row_idx, e_rt_col_idx)
        self.keypoint_dict['fruit_body'] = (fruit_tensor > 0).nonzero()
        
        if self.verbose:
            print("Keypoints info:\n",self.keypoint_dict)  
        print("TODO: May need to also pass the all envelope pixels")

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
    
    def object_mask_filter(self, semantic_dict, seg_masks):
        """
        Perform picking condition check 
        :param semantic_dict, masks
        :param masks
        :return: picking_condi:  
                 segmentation_mask: (2,w,h)
        """
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
                print("TODO: max fushion(later), right now I just assign overlap pixels to fruit")
                overlap_indices = overlap_tensor.nonzero(as_tuple=False)
                for idx in overlap_indices:
                    peduncle_mask[idx[0], idx[1]] = 0    # Assign the overlap pixels to fruit: Set 0 using the indices in peduncle_mask
                filtered_mask = torch.stack([avo_mask, peduncle_mask])
                return True, filtered_mask
            else:
                rospy.loginfo(f"Peduncle and fruit are both visiable, but not attaching, you may have occlusion in between or wrong detection")
                return False, seg_masks
    
    def semantic_augmenation(self, fruit_contour,fruit_envelope,seg_filtered_masks,semantic_dict):
        # Augment: add label 2 as contour and assign condfi score equal to fruit
        semantic_dict[2.0] = semantic_dict[0.0]
        # Augment: add label 3 as envelope and assign condfi score equal to fruit
        semantic_dict[3.0] = semantic_dict[0.0]
        
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
        seg_masks = results[0].masks.data.cpu()                                     # tensor cpu torch.Size([1, 640, 640])
        
        if len(semantic_res) ==0:
            print("Network detected nothing. Please recheck")
            return None, None, None, None
        else:
            # Filtering duplicated class id, retain the id with higher conf score
            semantic_dict, seg_masks = self.object_class_filter(semantic_res, seg_masks)
            
            # Filter object mask overlapping issue and check picking condition 
            picking_condi, seg_filtered_masks  = self.object_mask_filter(semantic_dict, seg_masks)
        
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
        
            # Calculate fruit keypoints
            self.set_fruit_keypoints(fruit_contour, fruit_envelope, seg_filtered_masks)
        
            # TODO: May not need this: Calculate 3D line from peduncle
            self.set_3dline_from_peduncle(fruit_contour)
        return annotated_image, seg_masks_aug, semantic_aug_dict,picking_condi



    #TODO: get the 3dline from peduncle
    def set_3dline_from_peduncle(self,fruit_mask_contour):
        self.line_3d = None

    #TODO: May not need. Delete this function once project is done
    def generate_square_mask(self):
        # Define the dimensions of the image
        image_size = (10, 10)

        # Create a 10x10 tensor initialized with zeros
        image_tensor = torch.zeros(image_size, dtype=torch.float32)

        # Calculate the start and end indices for the 4x4 square
        # The square should be centered, so for a 10x10 image with a 4x4 square:
        # Start at index (10 - 4)/2 = 3 and end at index 3 + 4 = 7
        start_index = (10 - 4) // 2
        end_index = start_index + 6

        # Set the pixels belonging to the square to 1
        image_tensor[start_index:end_index, start_index:end_index] = 1

        # Print the tensor to see the result
        print(image_tensor)
        
        return image_tensor

    #TODO: May not need. Delete this function once project is done
    def extract_fruit_contour_test(self, tensor):
        print(tensor.shape)
        # obj_nb, height, width = tensor.shape
        height, width = tensor.shape
        # tensor = tensor[0, ...]
        contour_tensor = torch.zeros((height, width), dtype=torch.float32)
        
        # Iterate over each pixel in the tensor
        for i in range(height):
            for j in range(width):
                # Check only pixels that are '1'
                if tensor[i, j] == 1:
                    # Initialize a flag to check boundary conditions
                    is_contour = False
                    
                    # Check if the current pixel is at the boundary of the tensor
                    if i == 0 or i == height-1 or j == 0 or j == width-1:
                        is_contour = True  # Consider boundary pixels as part of the contour
                    else:
                        # Check the neighbors (up, down, left, right)
                        if tensor[i-1, j] == 0 or tensor[i+1, j] == 0 or tensor[i, j-1] == 0 or tensor[i, j+1] == 0:
                            is_contour = True
                    
                    # If it's a contour (boundary or has zero neighbors), set it
                    if is_contour:
                        contour_tensor[i, j] = 1

        return contour_tensor
    
    #TODO: May not need. Delete this function once project is done
    def line_continuity(self, mask_tensor):
        line_continuity = True
        indices = (mask_tensor == 1).nonzero(as_tuple=False)
        # print(indices)
        total_idx= indices.shape[0]
        print("total dots:", total_idx)
        has_neighbor_cnt = 0
        isolated_count = 0
        # Loop through each index of '1' in the tensor
        for index in indices:
            print("index is ",index)
            i, j = index
            # has_neighbor = False
            local_neighbor_cnt= 0
            # Check all potential neighbors
            for di, dj in self.neighbors:
                # neighbor index
                ni, nj = i + di, j + dj
                # Check if the neighbor index is within bounds
                if 0 <= ni < mask_tensor.size(0) and 0 <= nj < mask_tensor.size(1):
                    # Check if there is a '1' in any neighbor
                    if mask_tensor[ni, nj] == 1:
                        # has_neighbor = True
                        local_neighbor_cnt += 1
                        # if local_neighbor_cnt>=2:
                        #     has_neighbor_cnt += 1
                        #     break
                        
            # If fewer than two neighbors with '1' were found, increase the isolated count
            if local_neighbor_cnt < 2:
                isolated_count += 1
                print("index is ",index)
                
            # If no neighbor with '1' was found, increase the isolated count
            # if not has_neighbor:
                # isolated_count += 1
        discontinuity_index = total_idx - has_neighbor_cnt
        print("discontinuity_index:", discontinuity_index)
        print("isolated_count:", isolated_count/2)
        # for now ignore this case:
                    #  0., 0., 1.
                    #  0., 1., 1.
                    #  0., 0., 0.
        return line_continuity
    
    #TODO: May not need. Delete this function once project is done
    def get_square_contour(self, image_tensor):
        # Get the dimensions of the image tensor
        height, width = image_tensor.shape
        # Initialize a contour tensor with the same shape, filled with zeros
        contour_tensor = torch.zeros((height, width), dtype=torch.float32)
        # Loop through each element in the image tensor to find the square's contour
        for i in range(height):
            for j in range(width):
                # Check if the current pixel is part of the square (value 1)
                if image_tensor[i, j] == 1:
                    # Check the neighbors (left, right, top, bottom)
                    if (i > 0 and image_tensor[i-1, j] == 0) or (i < height-1 and image_tensor[i+1, j] == 0) \
                    or (j > 0 and image_tensor[i, j-1] == 0) or (j < width-1 and image_tensor[i, j+1] == 0):
                        # If any neighbor is outside the square, mark this as a contour
                        contour_tensor[i, j] = 1
        return contour_tensor

















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
    