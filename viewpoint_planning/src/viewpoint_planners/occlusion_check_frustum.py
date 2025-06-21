import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import time

start = time.time()
# Shift function to move nodes to the center of the grid
def shift_to_center(coords):
    return coords + 0.5

# Updated camera position shifted to the center
camera_pos = shift_to_center(np.array([10, 1, 5]))

# Define multiple target positions shifted to the center
target_positions = [
    shift_to_center(np.array([10, 11, 5])),
    shift_to_center(np.array([10, 11, 6])),
    shift_to_center(np.array([11, 11, 6])),
    shift_to_center(np.array([11, 11, 5]))
]

# Define the ROI bounds
roi_min = np.array([5, 10, 5])
roi_max = np.array([15, 20, 15])

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the camera node in green
ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], color='green', label='Camera', s=100)

# Calculate the distances between the camera and each target
target_distances = [np.linalg.norm(target_pos - camera_pos) for target_pos in target_positions]

# Find the minimum distance (closest target)
min_target_distance = min(target_distances)

# Plot the target nodes in red
for target_pos in target_positions:
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='red', s=10)

    # Generate points along the ray (before and after target)
    num_points_before_target = 100
    num_points_after_target = 50

    # Points before hitting the target (in grey)
    points_before_target = np.linspace(camera_pos, target_pos, num_points_before_target)
    ax.plot(points_before_target[:, 0], points_before_target[:, 1], points_before_target[:, 2], color='grey', lw=2)

    # Points after hitting the target (in yellow)
    max_distance_after_target = 20
    direction_vector = target_pos - camera_pos
    direction_vector /= np.linalg.norm(direction_vector)
    all_points_after_target = target_pos + np.outer(np.linspace(0, max_distance_after_target, num_points_after_target), direction_vector)
    ax.plot(all_points_after_target[:, 0], all_points_after_target[:, 1], all_points_after_target[:, 2], color='yellow', lw=2)

# Generate all points inside the ROI and shift them to the center
roi_points = np.array(list(product(np.arange(roi_min[0], roi_max[0] + 1),
                                   np.arange(roi_min[1], roi_max[1] + 1),
                                   np.arange(roi_min[2], roi_max[2] + 1))))
shifted_roi_points = shift_to_center(roi_points)

# Calculate distances of all ROI points to the camera
roi_distances = np.linalg.norm(shifted_roi_points - camera_pos, axis=1)

# Plot all the nodes inside the ROI that are closer than the closest target
for i, point in enumerate(shifted_roi_points):
    if roi_distances[i] < min_target_distance:
        ax.scatter(*point, color='pink', s=10, alpha=0.5)
    else:
        ax.scatter(*point, color='grey', s=10, alpha=0.1)

# Draw ROI bounds as a cube
r = [roi_min, roi_max]
vertices = np.array(list(product([r[0][0], r[1][0]], [r[0][1], r[1][1]], [r[0][2], r[1][2]])))
for s, e in combinations(vertices, 2):
    if np.sum(np.abs(s - e)) == (roi_max - roi_min).min():
        ax.plot3D(*zip(s, e), color="orange", lw=2)

# Set the limits and labels of the plot
ax.set_xlim([0, 25])
ax.set_ylim([0, 25])
ax.set_zlim([0, 25])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

# Show the plot
plt.show()
