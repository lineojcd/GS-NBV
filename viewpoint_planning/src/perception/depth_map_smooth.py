import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Create a synthetic depth map of size 640x640 with values between 0 and 1
np.random.seed(42)  # For reproducibility
depth_map = np.random.rand(640, 640)

# Introduce noise by randomly setting some depth values to zero
# 10% of the values will be set to zero
noise_mask = np.random.rand(640, 640) < 0.39  
depth_map_noisy = depth_map.copy()
depth_map_noisy[noise_mask] = 0

noise_mask2 = np.random.rand(640, 640) > 0.4  
# depth_map_noisy = depth_map.copy()
depth_map_noisy[noise_mask2] += 0.2

# Apply Gaussian filter to smooth the noisy depth map
smoothed_depth_map = gaussian_filter(depth_map_noisy, sigma=5)

# Create a figure with two subplots
plt.figure(figsize=(12, 6))

# Plot the noisy depth map
plt.subplot(1, 2, 1)
plt.imshow(depth_map_noisy, cmap='jet', vmin=0, vmax=1)
plt.colorbar(label='Depth Value')
plt.title('Noisy Depth Map')

# Plot the smoothed depth map
plt.subplot(1, 2, 2)
plt.imshow(smoothed_depth_map, cmap='jet', vmin=0, vmax=1)
plt.colorbar(label='Depth Value')
plt.title('Smoothed Depth Map')

# Show the plots side by side
plt.show()
