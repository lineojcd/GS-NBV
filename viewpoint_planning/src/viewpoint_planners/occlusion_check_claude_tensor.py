import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

SHOW_PLOT = False

def distance(p1, p2):
    return torch.norm(p1 - p2)

def angle_between(v1, v2):
    v1_u = v1 / torch.norm(v1)
    v2_u = v2 / torch.norm(v2)
    return torch.acos(torch.clamp(torch.dot(v1_u, v2_u), -1.0, 1.0))

def is_behind_target(camera, target, point):
    v1 = target - camera
    v2 = point - camera
    return torch.dot(v1, v2) > torch.dot(v1, v1)

def shift_to_center(coords):
    return coords + 0.5


# filter_method = 'nearest'
filter_method = 'farthest'

start = time.time()

# Set up the cube in torch
x, y, z = torch.meshgrid(torch.arange(20), torch.arange(20), torch.arange(30))
cube = shift_to_center(torch.stack([x.ravel(), y.ravel(), z.ravel()], dim=1))

# Define camera and target positions (shifted to center)
camera = shift_to_center(torch.tensor([1, 3, 3], dtype=torch.float32))
targets = shift_to_center(torch.tensor([[5, 4, 4], [5, 4, 5], [7, 6, 6]], dtype=torch.float32))

# Find the farthest and nearest target
farthest_target = max(targets, key=lambda t: distance(camera, t))
print("farthest_target is", farthest_target)
nearest_target = min(targets, key=lambda t: distance(camera, t))
print("nearest_target is", nearest_target)
max_target_distance = distance(camera, farthest_target)
min_target_distance = distance(camera, nearest_target)

# Calculate centroid of targets
centroid = torch.mean(targets, dim=0)

if filter_method == 'nearest':
    # Identify points behind the nearest target
    behind_points = torch.stack([point for point in cube if distance(camera, point) > min_target_distance and is_behind_target(camera, nearest_target, point)])

if filter_method == 'farthest':
    behind_points = torch.stack([point for point in cube if distance(camera, point) > max_target_distance and is_behind_target(camera, farthest_target, point)])

# Calculate maximum and minimum angle
max_angle = max(angle_between(target - camera, centroid - camera) for target in targets)
min_angle = min(angle_between(target - camera, centroid - camera) for target in targets)

# Filter points based on the angle criterion
invisible = torch.stack([point for point in behind_points if angle_between(point - camera, centroid - camera) <= min_angle])

invisible_likely = torch.stack([point for point in behind_points if min_angle < angle_between(point - camera, centroid - camera) <= max_angle])

end = time.time()
dur = end - start
print("dur is", dur)

if SHOW_PLOT:
    # Create the plot (this part remains unchanged)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the cube nodes
    ax.scatter(cube[:, 0].numpy(), cube[:, 1].numpy(), cube[:, 2].numpy(), c='gray', alpha=0.1, s=10)
    # Plot invisible nodes
    ax.scatter(invisible[:, 0].numpy(), invisible[:, 1].numpy(), invisible[:, 2].numpy(), c='blue', s=20, label='Invisible Nodes')
    ax.scatter(invisible_likely[:, 0].numpy(), invisible_likely[:, 1].numpy(), invisible_likely[:, 2].numpy(), c='orange', s=20, label='Invisible Nodes')
    # Plot camera node
    ax.scatter(camera[0].item(), camera[1].item(), camera[2].item(), c='green', s=100, label='Camera')
    # Plot target nodes
    ax.scatter(targets[:, 0].numpy(), targets[:, 1].numpy(), targets[:, 2].numpy(), c='pink', s=100, label='Targets')
    # Plot centroid
    ax.scatter(centroid[0].item(), centroid[1].item(), centroid[2].item(), c='purple', s=150, label='Centroid')
    # Plot rays to targets and centroid
    for target in targets:
        ax.plot([camera[0].item(), target[0].item()], [camera[1].item(), target[1].item()], [camera[2].item(), target[2].item()], 'r--', alpha=0.3)
    ax.plot([camera[0].item(), centroid[0].item()], [camera[1].item(), centroid[1].item()], [camera[2].item(), centroid[2].item()], 'purple', linestyle='--', alpha=0.5)
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
