import sys
# sys.path.append('~/gradient_nbv_ws/src/gradientnbv/src/viewpoint_planning/src/perception/ultralytics_yolov8/')
# print(sys.path)
from ultralytics import YOLO
from PIL import Image
import cv2
import torch

# Load a pretrained YOLO model (recommended for training)
# model = YOLO("weights/yolov8m-seg.pt")
model = YOLO("weights/myvpp_sim_best.pt")
conf_threshold= 0.7

# Define path to the image file
# source = "images/bus.jpg"
# source = "images/myvpp_sim.v1i.yolov8/a_140.png"
source = "images/integration/test_p1.png"
model_path = "/home/jcd/gsnbv_ws/src/viewpoint_planning/src/perception/ultralytics_yolov8/weights/myvpp_sim_best.pt"
source = "images/integration/pictt.png"

# Perform object detection on an image using the model
results = model(source,conf= conf_threshold)
# results = model(source)
# for result in results:
#         print("result is :",result)


# for detection in results[0]:
#     class_index = detection.cls  # Get the class index of the detection
#     print("class_index is :", class_index)


# # Define a confidence threshold
# confidence_threshold = 0.5

# # Filter detections based on confidence
# filtered_boxes = [box for box in results.boxes if box.conf >= confidence_threshold]

# # Now, 'filtered_boxes' contains only the detections with confidence >= threshold
# for box in filtered_boxes:
#     print(f"Box: {box.xyxy}, Confidence: {box.conf}")

#     # Show or process your results as needed
#     # For example, showing the result:
#     box.show()

# # Save the result to disk if needed
# results.save(filename="images/integration/restest_p2_filtered.png")



# Process results list
# print(len(results))
# if False:
if True:
    for result in results:
        print("result is :",result)
        myres = result.masks
        # print("masks is :",result.masks)
        # print("masks is :",myres)
        print("masks data is :",myres.data)
        print("masks data shape is :",myres.data.shape)  #masks data shape is : torch.Size([1, 512, 640])
        # print("masks data is :",myres.data)
        total_sum = torch.sum(myres.data)
        print("Total sum:", total_sum)
        print("count :", total_sum/255)
        
        # Find the maximum value
        max_value = myres.data.max()
        print("Maximum value:", max_value)

        # Find the minimum value
        min_value = myres.data.min()
        print("Minimum value:", min_value)
        
        # Maximum value: tensor(1.)
        # Minimum value: tensor(0.)
        
        # original_image = cv2.imread(source)
        # segmentation_masks = result.masks.data
        # mask = (segmentation_masks > 0).astype('uint8') * 255  # Assuming binary mask needed
        # mask = mask[0]  # Taking the first mask if multiple are returned
        # mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        # result.show()  # display to screen
        # myres_box = result.boxes
        # print("myres_box is :",myres_box)
        
        # AttributeError: can't set attribute
        threshold = 0.5
        # mymask = myres_box.conf>threshold
        # myres_box.cls = myres_box.cls[mymask]
        # myres_box.conf = myres_box.conf[mymask]
        # myres_box.data = myres_box.data[mymask]
        # myres_box.xywh = myres_box.xywh[mymask]
        # myres_box.xywhn = myres_box.xywhn[mymask]
        # myres_box.xyxy = myres_box.xyxy[mymask]
        # myres_box.xyxyn = myres_box.xyxyn[mymask]
                
        # class_id_list = result.masks.cls.cpu().numpy()
        class_id_list = result.boxes.cls.cpu().numpy()
        class_conf_list = result.boxes.conf.cpu().numpy()
        semantic_res = list(zip(class_id_list, class_conf_list))
        print("semantic_res is :",semantic_res)
        
        class_id_list = result.boxes.cls.numpy()
        class_conf_list = result.boxes.conf.numpy()
        semantic_res = list(zip(class_id_list, class_conf_list))
        print("semantic_res2 is :",semantic_res)

        # print("result is :", result.boxes.cls.cpu().numpy())
        # exit(0)
        
        # boxes = result.boxes  # Boxes object for bounding box outputs
        # print("class_id = ",boxes.cls )
        
        # class_id = int(result.cls)  # Get class ID
        # class_id = result.cls  # Get class ID
        # class_label = result.names[class_id]  # Get class label from class ID
        # print(f'Detected class: {class_label}')  # Print class label
        
        
        
        # masks = result.masks  # Masks object for segmentation masks outputs
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # probs = result.probs  # Probs object for classification outputs
        # # print("probs=",probs)
        # # print("result.names=",result.names)
        # obb = result.obb  # Oriented boxes object for OBB outputs
        
        
        # result.save(filename="images/integration/restest_p5.png")  # save to disk