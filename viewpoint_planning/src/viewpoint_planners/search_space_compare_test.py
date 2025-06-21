import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r = 0.21
# Create a meshgrid for spherical coordinates for a quarter sphere
phi = np.linspace(0, np.pi/2 , 20)  # theta goes from 0 to pi/2 (quarter sphere)
theta = np.linspace(-0.25*np.pi, 1.25*np.pi , 500)  # phi goes from 0 to pi/2 (quarter of the circle)
theta, phi = np.meshgrid(theta,phi)
gs_theta = np.linspace(-0.25*np.pi, 1.25*np.pi , 500)

# Parametric equations for the coordinates of the sphere
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# Circle points
x_circle = r * np.cos(theta)  # x = cos(theta) for unit circle
y_circle = r * np.sin(theta)  # y = sin(theta) for unit circle
z_circle = np.zeros_like(theta)  # z = 0 for all points (x-y plane)

# Create a figure for plotting the quarter sphere using points
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Draw lines
ax.plot([0, 0], [0, 0], [0, 2], color='b', linewidth=1, label='Z-axis')  # Red line from [0,0,0] to [0,0,1]
ax.plot([0, 0], [0, 2], [0, 0], color='g', linewidth=1, label='Y-axis')  # Green line from [0,0,0] to [0,1,0]
ax.plot([0, 2], [0, 0], [0, 0], color='r', linewidth=1, label='X-axis')  # Blue line from [0,0,0] to [1,0,0]

# Scatter plot using the points instead of a surface
# ax.scatter(x, y, z, color='cyan',  label='SC-NBVP search space')  # s controls the size of the points
# ax.scatter(x, y, z, color='cyan', s=10, label='SC-NBVP search space')  # s controls the size of the points
# Plot the surface
ax.plot_surface(x, y, z, color='cyan', alpha=1, rstride=50, cstride=1, label='SC-NBVP search space')

# Scatter plot for the circle
ax.scatter(x_circle, y_circle, z_circle, color='m', s=15, label='GS-NBV search space')

# Set the axis labels
ax.set_xlabel('X/m')
ax.set_ylabel('Y/m')
ax.set_zlabel('Z/m')

# Set the axis limits
ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-0.25, 0.25])
ax.set_zlim([-0.25, 0.25])

# Set custom ticks for x, y, and z axes
# ax.set_xticks([-1, -0.5, 0, 0.5, 1])
# ax.set_xticks([-1,  0,  1])
# ax.set_yticks([-1, -0.5, 0, 0.5, 1])
# ax.set_yticks([-1,  0,  1])
# ax.set_yticks([-1,  0])
# ax.set_zticks([-1, -0.5, 0, 0.5, 1])
# ax.set_zticks([-1,  0,  1])
# ax.set_zticks([  0,  1])

# Set the background color and make it appear lighter
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))  # Background of x plane
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))  # Background of y plane
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))  # Background of z plane

# Optionally turn off the grid lines to make it cleaner
# ax.grid(False)

# Set the aspect ratio for equal scaling
ax.set_box_aspect([1, 1, 1])

# Add a legend
ax.legend(loc='upper right')

# Show the plot
plt.show()





