import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

# Define the Cam rotation using a quaternion
r_cam = R.from_quat([0.5, -0.5, 0.5, 0.5])

# Define the GNBV rotation using Euler angles
r_gnbv = R.from_euler("xyz", [-np.pi / 2, 0.0, -np.pi / 2])

# Get the rotation matrices
rotation_matrix_cam = r_cam.as_matrix()
rotation_matrix_gnbv = r_gnbv.as_matrix()

# Define the origins of the frames
origin = np.zeros((3,))
origin_cam = np.array([0.2, 0.2, 0.2])
origin_gnbv = np.array([0.5, 0.5, 0.5])

# Define the coordinate axes in the original frame
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

# Rotate the coordinate axes using the rotation matrices
x_rot_cam = rotation_matrix_cam @ x_axis
y_rot_cam = rotation_matrix_cam @ y_axis
z_rot_cam = rotation_matrix_cam @ z_axis

x_rot_gnbv = rotation_matrix_gnbv @ x_axis
y_rot_gnbv = rotation_matrix_gnbv @ y_axis
z_rot_gnbv = rotation_matrix_gnbv @ z_axis

# Plot the original, Cam, and GNBV coordinate frames
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original (world) coordinate frame with dashed lines (no arrows)
ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='red', linestyle='--', label='World X (Dashed)')
ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='green', linestyle='--', label='World Y (Dashed)')
ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='blue', linestyle='--', label='World Z (Dashed)')

# Plot the Cam coordinate frame with its origin at (0.2, 0.2, 0.2)
ax.quiver(origin_cam[0], origin_cam[1], origin_cam[2], x_rot_cam[0], x_rot_cam[1], x_rot_cam[2], color='green', length=1, normalize=True, label='Cam X')
ax.quiver(origin_cam[0], origin_cam[1], origin_cam[2], y_rot_cam[0], y_rot_cam[1], y_rot_cam[2], color='blue', length=1, normalize=True, label='Cam Y')
ax.quiver(origin_cam[0], origin_cam[1], origin_cam[2], z_rot_cam[0], z_rot_cam[1], z_rot_cam[2], color='red', length=1, normalize=True, label='Cam Z')

# Plot the GNBV coordinate frame with its origin at (0.5, 0.5, 0.5)
ax.quiver(origin_gnbv[0], origin_gnbv[1], origin_gnbv[2], x_rot_gnbv[0], x_rot_gnbv[1], x_rot_gnbv[2], color='green', length=1, normalize=True, label='GNBV X')
ax.quiver(origin_gnbv[0], origin_gnbv[1], origin_gnbv[2], y_rot_gnbv[0], y_rot_gnbv[1], y_rot_gnbv[2], color='blue', length=1, normalize=True, label='GNBV Y')
ax.quiver(origin_gnbv[0], origin_gnbv[1], origin_gnbv[2], z_rot_gnbv[0], z_rot_gnbv[1], z_rot_gnbv[2], color='red', length=1, normalize=True, label='GNBV Z')

# Set plot labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1.5])
ax.set_ylim([-1, 1.5])
ax.set_zlim([-1, 1.5])

# Show the legend
ax.legend()

# Show the plot
plt.show()
