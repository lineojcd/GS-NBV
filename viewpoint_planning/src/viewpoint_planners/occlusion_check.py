import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import time

# Shift function to move nodes to the center of the grid
def shift_to_center(coords):
    return coords + 0.5

start = time.time()

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




# Plot the target nodes in red
for target_pos in target_positions:
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='red', s=10)

    # Calculate the direction vector of the ray from the camera to each target
    direction_vector = target_pos - camera_pos
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize the direction

    # Generate points along the ray (before and after target)
    num_points_before_target = 100
    num_points_after_target = 50

    # Points before hitting the target (in grey)
    points_before_target = np.linspace(camera_pos, target_pos, num_points_before_target)

    # Points after hitting the target (we'll filter out those outside the ROI)
    max_distance_after_target = 20  # Define the distance you want to go after the target
    all_points_after_target = np.array([target_pos + i * direction_vector for i in np.linspace(0, max_distance_after_target, num_points_after_target)])

    # Filter the points after the target to only include those within the ROI bounds
    points_in_roi = np.array([point for point in all_points_after_target 
                              if all(roi_min <= point) and all(point <= roi_max)])

    # Plot the points before the target (in grey)
    ax.plot(points_before_target[:, 0], points_before_target[:, 1], points_before_target[:, 2], color='grey', lw=2)

    # Plot the points after the target (in yellow) that are inside the ROI
    if len(points_in_roi) > 0:
        ax.scatter(points_in_roi[:, 0], points_in_roi[:, 1], points_in_roi[:, 2], color='yellow', s=10)


# Generate all points inside the ROI and shift them to the center
roi_grid_x = np.arange(roi_min[0], roi_max[0] + 1)
roi_grid_y = np.arange(roi_min[1], roi_max[1] + 1)
roi_grid_z = np.arange(roi_min[2], roi_max[2] + 1)

roi_points = np.array(np.meshgrid(roi_grid_x, roi_grid_y, roi_grid_z)).T.reshape(-1, 3)
shifted_roi_points = shift_to_center(roi_points)

# Plot all the nodes inside the ROI (shifted) in grey, with alpha = 0.1
ax.scatter(shifted_roi_points[:, 0], shifted_roi_points[:, 1], shifted_roi_points[:, 2], color='grey', s=10, alpha=0.1)



# Visualize the ROI range as a cube
r = [roi_min, roi_max]

# Define the vertices of the ROI cube
vertices = np.array(list(product([r[0][0], r[1][0]], [r[0][1], r[1][1]], [r[0][2], r[1][2]])))

# Draw lines between vertices to form the cube (ROI)
for s, e in combinations(vertices, 2):
    if np.sum(np.abs(s - e)) == (roi_max - roi_min).min():  # Only connect points that form the edges
        ax.plot3D(*zip(s, e), color="orange", lw=2)

# Set the limits of the plot based on the 3D space
ax.set_xlim([0, 25])
ax.set_ylim([0, 25])
ax.set_zlim([0, 25])

# Set labels for the axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Add legend
ax.legend()

end = time.time()
duration = end - start 
print("duration is:", duration)

# Show the plot
plt.show()


