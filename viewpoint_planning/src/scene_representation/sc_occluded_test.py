# Reimport necessary libraries and rewrite the code
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Updated dimensions and camera/target nodes
x_limit, y_limit, z_limit = 8, 8, 8
# Updated camera and target nodes
camera_node = np.array([1, 3, 3]) + 0.5
target_node = np.array([5, 4, 4]) + 0.5  # Target node is now at (5, 4, 4)

# Calculate the vector from the camera to the target
ray_vector = target_node - camera_node
ray_direction = ray_vector / np.linalg.norm(ray_vector)

# Generate all possible nodes in the cube (centered in each voxel)
x_range = np.arange(0, x_limit) + 0.5
y_range = np.arange(0, y_limit) + 0.5
z_range = np.arange(0, z_limit) + 0.5

grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
nodes = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

# Function to check if a node is behind the target node
def is_behind_target(node, camera_node, target_node, ray_direction):
    vector_to_node = node - camera_node
    dot_product = np.dot(vector_to_node, ray_direction)
    return dot_product > np.dot(target_node - camera_node, ray_direction)

# Find unvisible nodes
unvisible_nodes = np.array([node for node in nodes if is_behind_target(node, camera_node, target_node, ray_direction)])

# Plot the camera node, target node, unvisible nodes, and ray
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the unvisible nodes in blue
if len(unvisible_nodes) > 0:
    ax.scatter(unvisible_nodes[:, 0], unvisible_nodes[:, 1], unvisible_nodes[:, 2], color='blue', label='Unvisible Nodes')

# Plot the target node in pink (centered in its voxel)
ax.scatter(*target_node, color='pink', s=100, label='Target Node')

# Plot the camera node in green (centered in its voxel)
ax.scatter(*camera_node, color='green', s=100, label='Camera Node')

# Draw the ray from camera to target node
ax.plot([camera_node[0], target_node[0]], [camera_node[1], target_node[1]], [camera_node[2], target_node[2]], color='red', label='Ray')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Set the limits to match the voxel grid dimensions
ax.set_xlim([0, x_limit])
ax.set_ylim([0, y_limit])
ax.set_zlim([0, z_limit])

# Ensure the tick marks are evenly spaced at unit intervals
ax.set_xticks(np.arange(0, x_limit + 1, 1))
ax.set_yticks(np.arange(0, y_limit + 1, 1))
ax.set_zticks(np.arange(0, z_limit + 1, 1))

# Show plot
plt.show()
