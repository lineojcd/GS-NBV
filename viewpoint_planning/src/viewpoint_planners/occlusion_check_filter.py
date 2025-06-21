import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_behind_target(camera, target, point):
    camera = np.array(camera)
    target = np.array(target)
    point = np.array(point)
    
    direction = target - camera
    t = np.dot(point - camera, direction) / np.dot(direction, direction)
    
    return t > 1

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the limits of the cube
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 15)

# Define camera and target positions
camera = (1, 3, 3)
target = (5, 4, 4)

# Plot camera (green) and target (pink) nodes
ax.scatter(*camera, color='green', s=100, label='Camera')
ax.scatter(*target, color='pink', s=100, label='Target')

# Plot the ray from camera to target
ax.plot([camera[0], target[0]], [camera[1], target[1]], [camera[2], target[2]], 'r--', label='Ray')

# Create a grid of points
x, y, z = np.meshgrid(np.arange(0.5, 10.5), np.arange(0.5, 10.5), np.arange(0.5, 15.5))

# Plot visible nodes
visible_nodes = np.array([(x, y, z) for x, y, z in zip(x.flatten(), y.flatten(), z.flatten())
                          if not is_behind_target(camera, target, (x, y, z))])
ax.scatter(visible_nodes[:, 0], visible_nodes[:, 1], visible_nodes[:, 2], color='gray', alpha=0.3)

# Plot invisible nodes
invisible_nodes = np.array([(x, y, z) for x, y, z in zip(x.flatten(), y.flatten(), z.flatten())
                            if is_behind_target(camera, target, (x, y, z))])
ax.scatter(invisible_nodes[:, 0], invisible_nodes[:, 1], invisible_nodes[:, 2], color='blue', alpha=0.3, label='Invisible Nodes')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Cube Visualization with Ray Tracing')

# Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()