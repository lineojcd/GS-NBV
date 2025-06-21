import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def is_behind_target(camera, target, point):
    v1 = np.array(target) - np.array(camera)
    v2 = np.array(point) - np.array(camera)
    return np.dot(v1, v2) > np.dot(v1, v1)

def shift_to_center(coords):
    return coords + 0.5




# filter_method = 'nearest'
filter_method = 'farthest'

start = time.time()
# Set up the cube
x, y, z = np.meshgrid(np.arange(20), np.arange(20), np.arange(30))
cube = shift_to_center(np.c_[x.ravel(), y.ravel(), z.ravel()])

# Define camera and target positions (shifted to center)
camera = shift_to_center(np.array([1, 3, 3]))
targets = shift_to_center(np.array([[5, 4, 4], [5, 4, 5], [7, 6, 6]]))

# Find the farthest target and calculate max distance
farthest_target = max(targets, key=lambda t: distance(camera, t))
print("farthest_target is", farthest_target)
nearest_target = min(targets, key=lambda t: distance(camera, t))
print("nearest_target is", nearest_target)
max_target_distance = distance(camera, farthest_target)
min_target_distance = distance(camera, nearest_target)

# Calculate centroid of targets
centroid = np.mean(targets, axis=0)

if filter_method == 'nearest':
    # First, identify points behind the farthest target
    behind_points = np.array([
        point for point in cube 
        if distance(camera, point) > min_target_distance and is_behind_target(camera, nearest_target, point) ])

if filter_method == 'farthest':
    behind_points = np.array([
    point for point in cube 
    if distance(camera, point) > max_target_distance and is_behind_target(camera, farthest_target, point) ])


# Calculate maximum and minimum angle
max_angle = max(angle_between(target - camera, centroid - camera) for target in targets)
min_angle = min(angle_between(target - camera, centroid - camera) for target in targets)

# Then, filter these points based on the angle criterion
invisible = np.array([
    point for point in behind_points
    if angle_between(point - camera, centroid - camera) <= min_angle
])

invisible_likely = np.array([
    point for point in behind_points
    if min_angle < angle_between(point - camera, centroid - camera) <= max_angle
])

end = time.time()
dur = end-start
print("dur is ", dur)



# Create the plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# Plot the cube nodes
ax.scatter(cube[:, 0], cube[:, 1], cube[:, 2], c='gray', alpha=0.1, s=10)
# Plot invisible nodes
ax.scatter(invisible[:, 0], invisible[:, 1], invisible[:, 2], c='blue', s=20, label='Invisible Nodes')
ax.scatter(invisible_likely[:, 0], invisible_likely[:, 1], invisible_likely[:, 2], c='orange', s=20, label='Invisible Nodes')
# Plot camera node
ax.scatter(*camera, c='green', s=100, label='Camera')
# Plot target nodes
ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='pink', s=100, label='Targets')
# Plot centroid
ax.scatter(*centroid, c='purple', s=150, label='Centroid')
# Plot rays to targets and centroid
for target in targets:
    ax.plot([camera[0], target[0]], [camera[1], target[1]], [camera[2], target[2]], 'r--', alpha=0.3)
ax.plot([camera[0], centroid[0]], [camera[1], centroid[1]], [camera[2], centroid[2]], 'purple', linestyle='--', alpha=0.5)
# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Cube Visualization with Centered Grid Points')
# Set axis limits
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 40)
# Add legend
ax.legend()
# Show the plot
plt.tight_layout()
plt.show()


if __name__ == '__main__':
    pass