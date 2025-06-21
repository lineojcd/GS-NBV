import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define points as NumPy arrays
center_offset = np.array([0.5, 0.5, 0.5]) 

# Test group: up right
point1_idx = np.array([0, 0, 0])  # Starting point
point2_idx = np.array([40, 60, 40])  # Ending point

# # Test group: up left
# point1_idx = np.array([0, 0, 0])  # Starting point
# point2_idx = np.array([-4, 6, 4])  # Ending point

# # Test group: bottom left
# point1_idx = np.array([0, 0, 0])  # Starting point
# point2_idx = np.array([-4, 6, -4])  # Ending point
 

# # Test group: straight front
# point1_idx = np.array([1, 0, 1])  # Starting point
# point2_idx = np.array([1, 4, 1])  # Ending point

point1 = point1_idx + center_offset
point2 = point2_idx + center_offset

# Create the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the given points
ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='red')
# Connect the points with a line
ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='blue')

# Set grid for visualization
ax.grid(True)

# Define the line equation using the points
direction = point2 - point1
line_points = np.linspace(point1, point2, num=10)

print("line_points: ",line_points)

# Calculate the steps needed to pass through each voxel
step_size = 0.3  # Small step size to ensure we catch all voxels
num_steps = int(np.linalg.norm(direction) / step_size)
steps = np.linspace(0, 1, num_steps)
print("num_steps is:",num_steps)
print("steps are:",steps)

# List to store highlighted voxel indices
highlighted_voxels = []

# Loop through steps to find all voxels the line passes through
for st in steps:
    current_point = point1 + st * direction
    voxel = tuple(np.floor(current_point).astype(int))
    if voxel not in highlighted_voxels:
        highlighted_voxels.append(voxel)

# Filter highlighted_indices
if (point1_idx[0], point1_idx[1], point1_idx[2] ) in highlighted_voxels:
    highlighted_voxels.remove((point1_idx[0], point1_idx[1], point1_idx[2]))
    print("Element removed:", point1_idx)
if (point2_idx[0], point2_idx[1], point2_idx[2] ) in highlighted_voxels:
    highlighted_voxels.remove((point2_idx[0], point2_idx[1], point2_idx[2]  ) )
    print("Element removed:", point2_idx)

# Draw the voxels the line passes through
for voxel in highlighted_voxels:
    x, y, z = voxel
    voxel_corners = np.array([[x, y, z],
                              [x+1, y, z],
                              [x+1, y+1, z],
                              [x, y+1, z],
                              [x, y, z+1],
                              [x+1, y, z+1],
                              [x+1, y+1, z+1],
                              [x, y+1, z+1]])
    faces = [[voxel_corners[j] for j in [0, 1, 2, 3]], # bottom
             [voxel_corners[j] for j in [4, 5, 6, 7]], # top
             [voxel_corners[j] for j in [0, 1, 5, 4]], # front
             [voxel_corners[j] for j in [2, 3, 7, 6]], # back
             [voxel_corners[j] for j in [1, 2, 6, 5]], # right
             [voxel_corners[j] for j in [4, 7, 3, 0]]] # left
    ax.add_collection3d(Poly3DCollection(faces, color='lightpink', alpha=0.2))

# Set axis limits to show integer grid lines more clearly
# ax.set_xticks(np.arange(min(point1_idx[0], point2_idx[0]), max(point1_idx[0], point2_idx[0]) + 1, 1))
# ax.set_yticks(np.arange(min(point1_idx[1], point2_idx[1]), max(point1_idx[1], point2_idx[1]) + 1, 1))
# ax.set_zticks(np.arange(min(point1_idx[2], point2_idx[2]), max(point1_idx[2], point2_idx[2]) + 1, 1))

# ax.set_xticks(np.arange(-5, 5, 1))
# ax.set_yticks(np.arange(-5, 5, 1))
# ax.set_zticks(np.arange(-5, 5, 1))


# ax.set_yticks(np.arange(min(point1_idx[1], point2_idx[1]), max(point1_idx[1], point2_idx[1]) + 1, 1))
# ax.set_zticks(np.arange(min(point1_idx[2], point2_idx[2]), max(point1_idx[2], point2_idx[2]) + 1, 1))

# Set labels and show plot
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Plot of Points and Connecting Line with Highlighted Voxels')

# Print the highlighted voxel indices
print("Highlighted voxel indices:", len(highlighted_voxels))
for voxel in highlighted_voxels:
    print(voxel)

plt.show()