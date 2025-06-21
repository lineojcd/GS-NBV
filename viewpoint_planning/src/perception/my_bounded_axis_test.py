import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_bounded_fruit_axis(computed_axis, bounded_angle = 30, plot = False, normalize=False):
    GT_axis = np.array([0, 0, 1])
    
    if normalize:
        # Step 1: Normalize the vectors
        GT_axis = GT_axis / np.linalg.norm(GT_axis)
        computed_axis = computed_axis / np.linalg.norm(computed_axis)

    # Step 2: Calculate the angle between the vectors
    cos_theta = np.dot(GT_axis, computed_axis)
    angle = np.degrees(np.arccos(cos_theta))

    print(f"Angle between GT_axis and computed_axis: {angle:.2f} degrees")

    # Initialize variables for visualization
    bounded_axis = None

    # Step 3: Check if the angle is greater than 45 degrees
    if angle > bounded_angle:
        # Calculate the rotation axis (cross product of GT_axis and computed_axis)
        rotation_axis = np.cross(GT_axis, computed_axis)
        
        # Normalize the rotation axis
        rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Calculate the axis that is 45 degrees from GT_axis
        angle_rad = np.radians(bounded_angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad) + (1 - np.cos(angle_rad)) * rotation_axis_normalized[0]**2,
            (1 - np.cos(angle_rad)) * rotation_axis_normalized[0] * rotation_axis_normalized[1] - np.sin(angle_rad) * rotation_axis_normalized[2],
            (1 - np.cos(angle_rad)) * rotation_axis_normalized[0] * rotation_axis_normalized[2] + np.sin(angle_rad) * rotation_axis_normalized[1]],
            [(1 - np.cos(angle_rad)) * rotation_axis_normalized[1] * rotation_axis_normalized[0] + np.sin(angle_rad) * rotation_axis_normalized[2],
            np.cos(angle_rad) + (1 - np.cos(angle_rad)) * rotation_axis_normalized[1]**2,
            (1 - np.cos(angle_rad)) * rotation_axis_normalized[1] * rotation_axis_normalized[2] - np.sin(angle_rad) * rotation_axis_normalized[0]],
            [(1 - np.cos(angle_rad)) * rotation_axis_normalized[2] * rotation_axis_normalized[0] - np.sin(angle_rad) * rotation_axis_normalized[1],
            (1 - np.cos(angle_rad)) * rotation_axis_normalized[2] * rotation_axis_normalized[1] + np.sin(angle_rad) * rotation_axis_normalized[0],
            np.cos(angle_rad) + (1 - np.cos(angle_rad)) * rotation_axis_normalized[2]**2]
        ])
        
        bounded_axis = np.dot(rotation_matrix, GT_axis)
        
        print(f"Axis at {bounded_angle} degrees from GT_axis: {bounded_axis}")

    else:
        print("The angle is not greater than bounded degrees.")
        bounded_axis = computed_axis

    if plot:
        # Visualization in 3D space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the world axes with dashed lines
        ax.plot([0, 2], [0, 0], [0, 0], color='red', linestyle='--', label='World X')
        ax.plot([0, 0], [0, 2], [0, 0], color='green', linestyle='--', label='World Y')
        ax.plot([0, 0], [0, 0], [0, 2], color='blue', linestyle='--', label='World Z')

        # Plot GT_axis in magenta
        ax.quiver(0, 0, 0, GT_axis[0], GT_axis[1], GT_axis[2], color='magenta', label='GT_axis')

        # Plot computed_axis in black
        ax.quiver(0, 0, 0, computed_axis[0], computed_axis[1], computed_axis[2], color='black', label='computed_axis')

        # Plot bounded_axis in lime if it exists
        if bounded_axis is not None:
            ax.quiver(0, 0, 0, bounded_axis[0], bounded_axis[1], bounded_axis[2], color='lime', label='bounded_axis')

        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])

        # Show the legend
        ax.legend()

        # Show the plot
        plt.show()
    
    return bounded_axis


if __name__ == '__main__':
    # Given vectors
    computed_axis = np.array([-0.0059408, 0.94147, 0.33705])
    bounded_axis = get_bounded_fruit_axis(computed_axis, plot = True)
    print("bounded_axis is ", bounded_axis)
