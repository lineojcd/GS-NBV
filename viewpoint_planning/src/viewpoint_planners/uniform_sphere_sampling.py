import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to uniformly sample points between two concentric spheres (radii 1 and 2)
def sample_points_between_spheres(num_points, r_min=1, r_max=2):
    theta = np.random.uniform(-0.25 * np.pi, 1.25 * np.pi, num_points)  
    print("thetha min and max are ", np.min(theta), np.max(theta))
    phi = np.random.uniform(0, np.pi, num_points)  # Random azimuthal angles (0 to 2pi)
    r = np.random.uniform(r_min, r_max, num_points)  # Random radius between 1 and 2
    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    print("x y z:",x,y,z)
    return x, y, z

# Sample 500 points between the two spheres
num_points = 5000
x, y, z = sample_points_between_spheres(num_points)
points = np.stack((x, y, z), axis=-1)
print(points)
# Visualize the points between the two spheres using matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the points between the spheres
ax.scatter(x, y, z, color='g', s=10)

# Set labels and aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1, 1, 1])

# Show the plot
plt.show()
