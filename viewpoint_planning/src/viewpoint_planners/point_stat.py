# Importing matplotlib.pyplot since it was missed earlier
import matplotlib.pyplot as plt
import torch

# Generate 100 random points within the range [-1, 1] in 3D space
random_points = 2 * torch.rand(1500, 3) - 1  # Generates points between [-1, 1] for each coordinate

# Calculate the mean and standard deviation along the points
mean_random = random_points.mean(dim=0, keepdim=True)
std_random = random_points.std(dim=0, keepdim=True)

# Find points that lie within 1 standard deviation for each coordinate (x, y, z)
within_std_mask_random = (random_points >= (mean_random - std_random)) & (random_points <= (mean_random + std_random))

# Keep only the points that satisfy the condition for all three coordinates
within_std_mask_random_all = within_std_mask_random.all(dim=1)

# Separate points within and outside 1 std
points_within_std_random = random_points[within_std_mask_random_all]
points_outside_std_random = random_points[~within_std_mask_random_all]


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points within 1 std (red)
ax.scatter(points_within_std_random[:, 0], points_within_std_random[:, 1], points_within_std_random[:, 2], color='red', label='Within 1 std')

# Plot points outside 1 std (blue)
ax.scatter(points_outside_std_random[:, 0], points_outside_std_random[:, 1], points_outside_std_random[:, 2], color='blue', alpha=0.1, label='Outside 1 std')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Legend
ax.legend()

# Show plot
plt.show()
