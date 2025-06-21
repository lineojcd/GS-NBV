import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

num_phi=40 * 2
num_theta = 100
# Create mesh grids for spherical coordinates for a quarter sphere
phi = np.linspace(0, np.pi, num_phi)  # phi from 0 to pi/2 (quarter sphere)
theta = np.linspace(-0.25 * np.pi, 1.25 * np.pi, num_theta)  # theta from -0.25*pi to 1.25*pi
phi, theta = np.meshgrid(phi, theta)  # Meshgrid for proper parameterization

# Radii for the spheres
radii = np.arange(0.21, 0.26, 0.005)  # radii from 0.21 to 0.25 with step size 0.05
alphalist =  np.linspace(1, 0 , len(radii))
# Create a figure for plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Draw axes lines with labels
ax.plot([0, 0], [0, 0], [0, 0.3], color='b', linewidth=1)
ax.plot([0, 0], [0, 0.3], [0, 0], color='g', linewidth=1)
ax.plot([0, 0.3], [0, 0], [0, 0], color='r', linewidth=1)

# Calculate color based on theta for cmap
theta_values = (theta.flatten() + 0.25 * np.pi) / (1.5 * np.pi)  # Normalized theta values for coloring
colors = plt.cm.coolwarm(theta_values)  # Applying cmap
half_rows = colors[:int(len(colors)/2)]
reversed_half_rows = np.flip(half_rows, axis=0)
colors = np.vstack((reversed_half_rows, half_rows))

# Generate and plot spherical points for each radius with color gradient
SCNBV_SHOW = False
SCNBV_SHOW = True
if SCNBV_SHOW:
    for i, r in enumerate(radii):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        ax.scatter(x.flatten(), y.flatten(), z.flatten(), color=colors, s=1, alpha = alphalist[i])
        if r == 0.21:
            ax.plot_surface(x, y, z, color='blue', alpha=0.1, rstride=50, cstride=1, label='SC-NBVP search space 2D')
            ax.scatter(0,0,0.21, color='b', s=50, alpha=1, label='SC-NBVP search space 3D')
        # if r == 0.26:
            # ax.plot_surface(x, y, z, color=colors[int(len(colors)/2)], alpha=0.1, rstride=50, cstride=1, label='SC-NBVP search space 3D')



# Circle points on the x-y plane
samples_circle = 500
x_circle = radii[0] * np.cos(np.linspace(-0.25 * np.pi, 1.25 * np.pi, samples_circle))
y_circle = radii[0] * np.sin(np.linspace(-0.25 * np.pi, 1.25 * np.pi, samples_circle))
z_circle = np.zeros(samples_circle)

# Scatter plot for the circle
ax.plot(x_circle, y_circle, z_circle, color='m', linewidth=3, label='GS-NBV search space 1D')
# ax.scatter(x_circle, y_circle, z_circle, color='m', s=10, label='GS-NBV search space')



# Draw for GNBV
# Define the new vertices of the cuboid
vertices = np.array([
    [-0.26, 0.26, 0.21],    # Vertex 0
    [-0.26, -0.148, 0.21],  # Vertex 1
    [0.26, -0.148, 0.21],   # Vertex 2
    [0.26, 0.26, 0.21],     # Vertex 3
    [-0.26, 0.26, -0.11],   # Vertex 4
    [-0.26, -0.148, -0.11], # Vertex 5
    [0.26, -0.148, -0.11],  # Vertex 6
    [0.26, 0.26, -0.11]     # Vertex 7
])
# Generate the list of sides' polygons
faces = [
    [vertices[j] for j in [0, 1, 2, 3]],  # Top face
    [vertices[j] for j in [4, 5, 6, 7]],  # Bottom face
    [vertices[j] for j in [0, 4, 7, 3]],  # Side face
    [vertices[j] for j in [1, 5, 6, 2]],  # Side face
    [vertices[j] for j in [0, 1, 5, 4]],  # Side face
    [vertices[j] for j in [2, 3, 7, 6]]   # Side face
]
# Plot each face with a different color
colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange']
colors = 'orange'
cube_alpha = .05
for i, face in enumerate(faces):
    x, y, z = zip(*face)
    if i == 0:
        print(x,y,z)
        a,b,c = (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
        ax.add_collection3d(Poly3DCollection([list(zip(a,b,c))], facecolors=colors, linewidths=0.00, edgecolors='green', alpha=0.2, label='GNBV search space 3D'))
        ax.add_collection3d(Poly3DCollection([list(zip(x, y, z))], facecolors=colors, linewidths=0.00, edgecolors='green', alpha=cube_alpha))
    ax.add_collection3d(Poly3DCollection([list(zip(x, y, z))], facecolors=colors, linewidths=0.00, edgecolors='green', alpha=cube_alpha))
    


ax.set_xlabel('X/m')
ax.set_ylabel('Y/m')
ax.set_zlabel('Z/m')
ax.set_xlim([-0.26, 0.26])  # Set axis limits
ax.set_ylim([-0.26, 0.26])
ax.set_zlim([-0.26, 0.26])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0))  # Background of x y z plane
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.xaxis.pane.fill = False  # Remove the pane backgrounds
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.set_box_aspect([1, 1, 1])                            # Set the aspect ratio for equal scaling
# ax.legend(loc='upper right', fontsize='xx-large')       # Add a legend
ax.legend(loc='right', fontsize='xx-large')       # Add a legend
# Make background transparent
ax.set_facecolor('none')  # Removes the axes background
fig.patch.set_facecolor('none')  # Removes the figure background
plt.show()          # Show the plot
plt.savefig("arc_plot.png")  # Saves the plot as a PNG file

