import time
import torch
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
m1_tensor = torch.tensor(
    [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
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
m2_tensor = torch.tensor(
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

meta_tensor = torch.tensor(
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

np_tensor = meta_tensor.numpy()
print("meta_tensor is \n", meta_tensor)
morph_kernel = 4
structure_erosion = np.ones((morph_kernel, morph_kernel), dtype=bool)
structure_dilation_c = np.ones((morph_kernel, morph_kernel), dtype=bool)

morph_kernel_2 = 2
structure_dilation_c2 = np.ones((morph_kernel_2, morph_kernel_2), dtype=bool)

env_morph_kernel_e = 3
env_morph_kernel_e2 = 5
structure_dilation_e = np.ones((env_morph_kernel_e, env_morph_kernel_e), dtype=bool)
structure_dilation_e2 = np.ones((env_morph_kernel_e2, env_morph_kernel_e2), dtype=bool)

# Apply erosion and then dilation (morphological opening)
eroded = binary_erosion(np_tensor, structure=structure_erosion)
# print("eroded is ", eroded)
opened_c = binary_dilation(eroded, structure=structure_dilation_c)
opened_c2 = binary_dilation(eroded, structure=structure_dilation_c2)
opened_e = binary_dilation(opened_c, structure=structure_dilation_e)
opened_e2 = binary_dilation(opened_c, structure=structure_dilation_e2)


# Convert back to PyTorch tensor
tensor_c = torch.tensor(opened_c, dtype=torch.float32)
tensor_c2 = torch.tensor(opened_c2, dtype=torch.float32)
tensor_e = torch.tensor(opened_e, dtype=torch.float32)
tensor_e2 = torch.tensor(opened_e2, dtype=torch.float32)

contour_res = tensor_c + tensor_c2
# Create a mask where the value is 1
contour_mask = (contour_res == 1)*1

contour_mask_2 = tensor_c - tensor_c2
envelope_mask = tensor_e2 - tensor_e

print("tensor_c is \n", tensor_c)
print("tensor_e is \n", tensor_e)
# print("tensor_c2 is \n", tensor_c2)
# print("contour_res is \n", contour_res)
# print("contour_mask is \n", contour_mask)
print("contour_mask_2 is \n", contour_mask_2)
# print(contour_mask_2 - contour_mask)
print("envelope_mask is \n", envelope_mask*3)

