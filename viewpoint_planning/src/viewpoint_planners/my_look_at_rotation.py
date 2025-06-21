import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def my_look_at_rotation(camera_position, target_position, plot = False, verbose=False):

    # Step 2: Compute the direction vector (Z-axis of the camera)
    direction_vector = target_position - camera_position
    direction_vector /= np.linalg.norm(direction_vector)  # Normalize the vector

    # Step 3: Create the rotation matrix
    z_axis = direction_vector  # Z-axis of the camera

    # Assume the global upward direction (initial X-axis of the camera)
    global_up = np.array([0, 0, 1])

    # Compute the Y-axis of the camera (right direction)
    y_axis = np.cross(z_axis, global_up)
    y_axis /= np.linalg.norm(y_axis)  # Normalize the Y-axis

    # Recompute the X-axis to ensure orthogonality
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)  # Normalize the X-axis

    
    # Construct the rotation matrix from the camera's axes
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    
    # Convert the rotation matrix to a quaternion
    rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()
    
    if verbose:
        # Print the rotation matrix and quaternion
        print("Rotation Matrix:\n", rotation_matrix)
        print("Rotation Quaternion:", rotation_quaternion)
    
    if plot:
        # Step 4: Plot the camera's coordinate frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the camera's position
        ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='red', s=100, label='Camera Position')

        # Plot the camera's X, Y, Z axes with custom colors and labels
        scale = 0.5  # Length of the axes for visualization
        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                x_axis[0], x_axis[1], x_axis[2], color='green', length=scale, normalize=True, label='Cam X-axis')
        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                y_axis[0], y_axis[1], y_axis[2], color='blue', length=scale, normalize=True, label='Cam Y-axis')
        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                z_axis[0], z_axis[1], z_axis[2], color='red', length=scale, normalize=True, label='Cam Z-axis')

        # Plot the target position
        ax.scatter(target_position[0], target_position[1], target_position[2], color='orange', s=100, label='Target Position')

        # Plot the line from the camera to the target
        ax.plot([camera_position[0], target_position[0]], 
                [camera_position[1], target_position[1]], 
                [camera_position[2], target_position[2]], 
                color='purple', linestyle='--', label='View Direction')

        # Plot the standard basis vectors from the origin with dashed lines
        ax.plot([0, 1], [0, 0], [0, 0], color='red', linewidth=2, linestyle='--', label='World X-axis')
        ax.plot([0, 0], [0, 1], [0, 0], color='green', linewidth=2, linestyle='--', label='World Y-axis')
        ax.plot([0, 0], [0, 0], [0, 1], color='blue', linewidth=2, linestyle='--', label='World Z-axis')

        # Set plot labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])

        ax.legend()

        plt.show()
    return rotation_quaternion
    
if __name__ == '__main__':
    # Step 1: Define camera and target positions
    camera_position = np.array([0.0, 0.1, 0.3])  # Example position of the camera
    target_position = np.array([-0.5, 0.1, 0.3])  # Example point the camera is looking at
    
#     target_position = np.array([0.5, -0.4, 1.18-0.7])  # Example point the camera is looking at
#     init_tar_dist = 0.55 
#     camera_position = np.array([target_position[0], target_position[1] + init_tar_dist, target_position[2]])  # Example position of the camera
    my_look_at_rotation(camera_position, target_position)
    
