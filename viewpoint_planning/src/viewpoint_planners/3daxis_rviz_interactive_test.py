import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_axis(ax, start_point, end_point, color='r'):
    """Plot an axis between start_point and end_point in the given color."""
    ax.plot([start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [start_point[2], end_point[2]],
            color=color, linewidth=2)

def interactive_3d_visualizer():
    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set initial plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initial plot: the world frame axes (red, green, blue)
    plot_axis(ax, [0, 0, 0], [1, 0, 0], color='red')   # X-axis
    plot_axis(ax, [0, 0, 0], [0, 1, 0], color='green') # Y-axis
    plot_axis(ax, [0, 0, 0], [0, 0, 1], color='blue')  # Z-axis

    plt.ion()  # Interactive mode on
    plt.show()

    while True:
        try:
            # Get user input for the axis start and end points
            # start_input = input("Enter the start point (x, y, z) or 'q' to quit: ")
            # if start_input.lower() == 'q':
            #     break
            
            start_input= "0, 0, 0"
            start_point = list(map(float, start_input.split(',')))

            end_input = input("Enter the end point (x, y, z): ")
            end_point = list(map(float, end_input.split(',')))

            # Clear previous plots (except the world axes)
            ax.cla()
            plot_axis(ax, [0, 0, 0], [1, 0, 0], color='red')   # X-axis
            plot_axis(ax, [0, 0, 0], [0, 1, 0], color='green') # Y-axis
            plot_axis(ax, [0, 0, 0], [0, 0, 1], color='blue')  # Z-axis

            # Plot the user-defined axis
            plot_axis(ax, start_point, end_point, color='magenta')

            # Update plot limits dynamically if necessary
            ax.set_xlim([min(-1, start_point[0], end_point[0]), max(1, start_point[0], end_point[0])])
            ax.set_ylim([min(-1, start_point[1], end_point[1]), max(1, start_point[1], end_point[1])])
            ax.set_zlim([min(-1, start_point[2], end_point[2]), max(1, start_point[2], end_point[2])])

            # Redraw the plot with the new axis
            plt.draw()

        except ValueError:
            print("Invalid input. Please enter coordinates in the form 'x,y,z'.")

if __name__ == "__main__":
    interactive_3d_visualizer()
