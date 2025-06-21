import numpy as np
import torch
def log_odds(p):
    return np.log(p / (1 - p))


res_0 = log_odds(0.0)
res_2 = log_odds(0.2)  # 0
res_4 = log_odds(0.4)  # -0.4
res_5 = log_odds(0.5)  # 0
res_9 = log_odds(0.9)  # 2.2
res_99 = log_odds(0.99)  # 2.2
res_999 = log_odds(0.999)  # 2.2
res_9999 = log_odds(0.9999)  # 2.2
res_99999 = log_odds(0.99999)  # 2.2
# res_10 = log_odds(1)
print("res_0 is ",res_0)
print("res_2 is ",res_2)
print("res_4 is ",res_4)
print("res_5 is ",res_5)
print("res_9 is ",res_9)
print("res_99 is ",res_99)
print("res_999 is ",res_999)
print("res_9999 is ",res_9999)
print("res_99999 is ",res_99999)
# print("res_10 is ",res_10)



lst = []
for i in range(len(lst)):
    print(i)
    
    
# ----------------------------------------  Assign the semantic class_conf and class_id

# Assuming 'self.device' is predefined, usually 'cpu' or 'cuda'
device = 'cpu'  # replace 'cpu' with 'cuda' if using GPU
image_size = (2, 3)
score_mask = -0.4 * torch.ones(image_size, dtype=torch.float32, device=device)
label_mask = -1 * torch.ones(image_size, dtype=torch.float32, device=device)
class_conf_log_odds1 = 0.2
class_conf_log_odds2 = 2.2
log_odds_lst = [class_conf_log_odds1,class_conf_log_odds2]

class_id1=1
class_id2=2
class_idlst = [class_id1,class_id2]

# Create two segmentation masks with binary values
segmentation_mask1 = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.float32, device=device)
segmentation_mask2 = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32, device=device)

# Create a list containing both segmentation masks
segmentation_mask_list = [segmentation_mask1, segmentation_mask2]

for obj in range(2):
    # Assign the semantic class_conf and class_id
    # 0: fruit; 1: peduncle; -1: other plant parts/background
    # Extract the class_id and the log odds of the class_conf
    class_id = class_idlst[obj]
    class_conf_log_odds = log_odds_lst[obj]
    
    # Create a mask for the current object where the segmentation mask is greater than 0
    current_mask = segmentation_mask_list[obj] > 0
    
    # Check where the new confidence is greater than what's already in the score_mask
    update_mask = (current_mask & (class_conf_log_odds > score_mask))
    
    label_mask[update_mask] = class_id
    score_mask[update_mask]  = class_conf_log_odds

print("label_mask Mask:", label_mask)
print("score_mask Mask:", score_mask)


# ----------------------------------------  Apply morphological operations 


# import torch
# import torch.nn.functional as F
# import torch
# import numpy as np
# from skimage.filters import median
# from skimage.morphology import disk

# # Your noisy segmentation tensor
# tensor = torch.tensor(
#     [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#      [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#      [0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
#      [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
#      [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
#      [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
#      [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
#      [1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
#      [1., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
#      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]
# )

# # Applying median filtering
# # Convert PyTorch tensor to NumPy array
# np_tensor = tensor.numpy()

# # Apply the median filter with a disk of radius 1
# filtered_tensor = median(np_tensor, disk(1))

# # Convert back to PyTorch tensor if needed
# filtered_tensor = torch.tensor(filtered_tensor)

# print(tensor)
# print(filtered_tensor)


import torch
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

tensor = torch.tensor(
    [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [1., 1., 0., 1., 1., 1., 1., 1., 1., 0.],
     [1., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]
)


tensor2 = torch.tensor(
    [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]
)

# Convert PyTorch tensor to NumPy array
np_tensor = tensor.numpy()

kernel_size=4
# Define the structure element for erosion and dilation
structure_erosion = np.ones((kernel_size, kernel_size), dtype=bool)
structure_dilation = np.ones((kernel_size, kernel_size), dtype=bool)

# Apply erosion and then dilation (morphological opening)
eroded = binary_erosion(np_tensor, structure=structure_erosion)
opened = binary_dilation(eroded, structure=structure_dilation)

# Convert back to PyTorch tensor
opened_tensor = torch.tensor(opened, dtype=torch.float32)


print(tensor)
print()
print(opened_tensor)


