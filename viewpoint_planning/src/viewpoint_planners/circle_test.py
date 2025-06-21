import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Iter 1
# if True:
#     def sample_points_on_circle(target_posi, axis, distance, num_samples=180):
#         axis = np.array(axis)
#         axis = axis / np.linalg.norm(axis)
#         orthogonal_vector = np.array([1, 0, 0])  # x-axis
#         second_vector = np.array([0, 1, 0])      # y-axis

#         theta = np.linspace(0, np.pi, num_samples)  # Half-circle from 0 to 180 degrees
#         points = np.array([target_posi + distance * (np.cos(t) * orthogonal_vector + np.sin(t) * second_vector) for t in theta])
#         print("points[0] is ",points[0])
#         return points, theta

#     target_position = np.array([0, 0, 0])
#     axis = np.array([0, 0, 1])
#     radius = 3
#     points, thetas = sample_points_on_circle(target_position, axis, radius)

#     extra_points = np.array([[1, 1, 0], [-1, 1, 0]])

#     # New point position
#     cam_position = np.array([0, 3, 0])

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Calculate intersection angles and find range
#     intersection_angles = []
#     for pt in extra_points:
#         direction = pt - target_position
#         angle = np.arctan2(direction[1], direction[0])
#         intersection_angles.append(angle)

#     min_angle = min(intersection_angles)
#     max_angle = max(intersection_angles)
#     print("min_angle and  max_angle are: ", min_angle, max_angle)
#     # Determine which points are within the intersection angle range
#     in_range = (thetas >= min_angle) & (thetas <= max_angle)
#     print("in_range are: ", in_range)

#     # Color points accordingly, exclude the first and last points from the "in range" classification
#     ax.scatter(points[in_range & (thetas > thetas[0]) & (thetas < thetas[-1]), 0], points[in_range & (thetas > thetas[0]) & (thetas < thetas[-1]), 1], points[in_range & (thetas > thetas[0]) & (thetas < thetas[-1]), 2], color='grey', label='Void points iter1')
#     ax.scatter(points[~in_range, 0], points[~in_range, 1], points[~in_range, 2], color='magenta', label='Valid Points iter1')
#     # Color the first point in Cyan and the last point in blue
#     ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='cyan', s=50, label='Start Point iter1')
#     ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='blue', s=50, label='End Point iter1')

#     ax.scatter(*target_position, color='red', s=100, label='Center iter1')

#     # New camera position
#     ax.scatter(*cam_position, color='maroon', s=50, label='Cam Pose iter1')

#     # Lines to the first and last sampled points
#     ax.plot([target_position[0], points[0, 0]], [target_position[1], points[0, 1]], [target_position[2], points[0, 2]], 'orange', linestyle='--', label='Line to Start Point iter1')
#     ax.plot([target_position[0], points[-1, 0]], [target_position[1], points[-1, 1]], [target_position[2], points[-1, 2]], 'orange', linestyle='--', label='Line to End Point iter1')

#     # Axis line (solid)
#     axis_line = np.array([target_position, target_position + axis])
#     ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], color='blue', linewidth=2, label='Axis Line iter1')


#     # Dictionary to hold all points of interest
#     points_dict = {
#         # "intersect_pt": extended_point.tolist(),  # Void points are the intersection range points
#         "intersect_pt": list(),                 # Void points are the intersection range points
#         "first_pt": points[0].tolist(),        # First point
#         "end_pt": points[-1].tolist(),         # End point
#         "cam_pt": cam_position.tolist(),       # Camera position
#         "obstacle_pt": extra_points.tolist()   # Given points (obstacles)
#     }

#     # Plotting the extra points and extending lines to their intersection with the circle
#     for i, pt in enumerate(extra_points):
#         ax.scatter(*pt, color='black', s=50, label='Given Point iter1' if i == 0 else "")
#         direction = pt - target_position
#         extended_point = target_position + direction / np.linalg.norm(direction) * radius
#         points_dict["intersect_pt"].append(extended_point.tolist())
#         if i == 0:
#             ax.plot([target_position[0], extended_point[0]], [target_position[1], extended_point[1]], [target_position[2], extended_point[2]], 'k--', label='Extended Line to Intersection iter1')
#         else:
#             ax.plot([target_position[0], extended_point[0]], [target_position[1], extended_point[1]], [target_position[2], extended_point[2]], 'k--')
#         ax.scatter(*extended_point, color='lime', s=50, label='Intersection Points close to Start iter1' if i == 0 else 'Intersection Points close to End iter1')

#     ax.set_aspect('auto')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     print(points_dict)
#     plt.show()
    














def sample_points_on_circle(target_posi, axis, distance, num_samples=180):
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    orthogonal_vector = np.array([1, 0, 0])  # x-axis
    second_vector = np.array([0, 1, 0])      # y-axis

    theta = np.linspace(0, np.pi, num_samples)  # Half-circle from 0 to 180 degrees
    points = np.array([target_posi + distance * (np.cos(t) * orthogonal_vector + np.sin(t) * second_vector) for t in theta])
    return points, theta

def find_closest_point(line_points, target_points):
    # Function to find closest points from a line to a set of target points
    closest_points = []
    for pt in target_points:
        distances = np.linalg.norm(line_points - pt, axis=1)
        closest_index = np.argmin(distances)
        closest_points.append(line_points[closest_index])
    return closest_points

# New center position for the additional circle
new_target_position = np.array([0, -0.3, 0])
axis = np.array([0, 0, 1])
radius = 3

# Sample points for the new circle
new_points, new_thetas = sample_points_on_circle(new_target_position, axis, radius)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the new circle
ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], color='orange', label='Sampled pts iter2')
# Mark the first and last point with different colors
ax.scatter(new_points[0, 0], new_points[0, 1], new_points[0, 2], color='cyan', label='First Pt iter2')
ax.scatter(new_points[-1, 0], new_points[-1, 1], new_points[-1, 2], color='blue', label='End Pt iter2')

# Points dictionary
points_dict = {
    'intersect_pt_f': [2.121, 2.121, 0.0],
    'intersect_pt_e': [-2.121, 2.121, 0.0],
    'first_pt': [3.0, 0.0, 0.0],
    'end_pt': [-3.0, 0, 0.0],
    'cam_pt': [0, 3, 0],
    'obstacle_pt': [[1, 1, 0], [-1, 1, 0]]
}

# Plotting points from the dictionary
obstacle_plotted = False
for key, pts in points_dict.items():
    if key == 'obstacle_pt':
        for pt in pts:
            ax.scatter(*pt, color='black', s=50, label='obstacle pt' if not obstacle_plotted else "")
            obstacle_plotted = True
            # Extend line to a new intersection point
            direction = np.array(pt) - new_target_position
            extended_point = new_target_position + direction / np.linalg.norm(direction) * radius
            ax.plot([new_target_position[0], extended_point[0]], [new_target_position[1], extended_point[1]], [new_target_position[2], extended_point[2]], 'k--', label='Extended Line to New Intersect' if not obstacle_plotted else "")
            ax.scatter(*extended_point, color='yellow', s=50, label='Inter_pt F iter2 OC' if not obstacle_plotted else 'Inter_pt E iter2 OC')
    else:
        ax.scatter(*pts, color='lime' if 'intersect' in key else 'red', s=50, label=f"{key.replace('_', ' ').title()} iter1")

# Connect intersect points to the circle and find intersection points
intersect_lines = [points_dict['intersect_pt_f'], points_dict['intersect_pt_e']]
intersect_points = find_closest_point(new_points, intersect_lines)
for i, pt in enumerate(intersect_points):
    label_suffix = 'F' if i == 0 else 'E'
    ax.scatter(*pt, color='green', s=50, label=f'Inter_pt {label_suffix} iter1 OC')
    ax.plot([new_target_position[0], pt[0]], [new_target_position[1], pt[1]], [new_target_position[2], pt[2]], 'purple', linestyle='--', label=f"Line to Intersect_pt {label_suffix} iter1")

# New camera position (iter2)
camera_posi_iter2 = [2.598, 1.5, 0]
ax.scatter(*camera_posi_iter2, color='purple', s=50, label='Cam Pt iter2')

# Axis line (solid)
axis_line = np.array([new_target_position, new_target_position + axis])
ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], color='blue', linewidth=2, label='Axis Line')

ax.set_aspect('auto')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()



# Reform this into While loop






