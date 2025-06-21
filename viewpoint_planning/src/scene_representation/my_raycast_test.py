import torch
import math

if False:
    # Create a 3D tensor with random values
    tensor = torch.randn(2, 3, 2)
    print("Original Tensor:\n", tensor)

    # Example 1: Using [:, -2:, :]
    sliced_tensor1 = tensor[:, -2:, :]
    print("Sliced Tensor using [:, -2:, :]:\n", sliced_tensor1)

    # Example 2: Using [..., -1, :]
    sliced_tensor2 = tensor[..., -1, :]
    print("Sliced Tensor using [..., -1, :]:\n", sliced_tensor2)


################2d line


# import matplotlib.pyplot as plt
# import numpy as np

# # Define points as NumPy arrays

# # positive slope line1
# point1 = np.array([-1, 0])      # Starting point
# point2 = np.array([4, 6])       # Ending point

# # positive slope line2
# # point1 = np.array([4, 6])     # Starting point
# # point2 = np.array([-1, 0])    # Ending point

# # negative slope line1
# # point1 = np.array([4, 0])  # Starting point has higher x-value
# # point2 = np.array([-1, 6])  # Ending point has lower x-value

# # negative slope line2
# # point1 = np.array([-1, 6])  # Starting point has higher x-value
# # point2 = np.array([4, 0])  # Ending point has lower x-value

# # vertical line1
# # point1 = np.array([0, 0])  # Starting point
# # point2 = np.array([0, 6])  # Ending point

# # vertical line2
# # point1 = np.array([0, 6])  # Starting point
# # point2 = np.array([0, 0])  # Ending point

# # Create the figure and axis
# fig, ax = plt.subplots()

# # Plot the given points
# ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], color='red')
# # Connect the points with a line
# ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='blue')
# # Set grid for visualization
# ax.grid(True)

# # Define the line equation y = mx + c using the points
# m = (point2[1] - point1[1]) / (point2[0] - point1[0])  # slope
# c = point1[1] - m * point1[0]  # y-intercept

# print("from pt", point1, "to", point2)
# print("slope is:", m )

# # List to store highlighted grid cell indices
# highlighted_indices = []
# if abs(m)  >  99999999:
#     print("vertical straight line")
#     x_range = range(point1[0], point2[0] + 1)
# if (m>0) and (m <99999999):
#     if point1[0] < point2[0]:
#         x_range = range(point1[0], point2[0])
#     if point1[0] > point2[0]:
#         x_range = range(point1[0], point2[0], -1)
# if (m<0) and (m > -99999999):
#     if point1[0] > point2[0]:
#         x_range = range(point1[0], point2[0], -1)
#     else:
#         x_range = range(point1[0], point2[0])

# print("x_range is ",x_range)
# # Loop over x range
# for ix in x_range:
#     print("ix is :", ix)
#     # Calculate exact y value at each integer x
#     y_at_x = m * ix + c
#     # Calculate y value at x+1
#     if (m>0) and (m <99999999):
#         if point1[0] < point2[0]:
#             y_at_x_next = m * (ix + 1) + c
#         if point1[0] > point2[0]:
#             y_at_x_next = m * (ix -1) + c
#     elif (m<0) and (m > -99999999):
#         if point1[0] > point2[0]:
#             y_at_x_next = m * (ix - 1) + c
#         else:
#             y_at_x_next = m * (ix + 1) + c
#     else:
#         y_at_x_next = m * ix + c
#     if abs(m)<  99999999:
#         # Calculate integer bounds for y
#         y_start = int(np.floor(min(y_at_x, y_at_x_next)))
#         y_end = int(np.ceil(max(y_at_x, y_at_x_next)))
#     else:
#         y_start , y_end = point1[1], point2[1]
#     print("y_at_x y_at_x_next  y_start y_end are: ",y_at_x,y_at_x_next,y_start,y_end)
#     if m>=0:
#         if point1[0] < point2[0]:
#             # Fill between y values at ix and ix+1, between current x and next x
#             ax.fill_betweenx([y_start, y_end], ix, ix + 1, color='yellow', alpha=0.5)
#         if point1[0] > point2[0]:
#             ax.fill_betweenx([y_start, y_end], ix, ix - 1, color='yellow', alpha=0.5)
            
        
#     else:
#         if point1[0] > point2[0]:
#             ax.fill_betweenx([y_start, y_end], ix, ix - 1, color='yellow', alpha=0.5)
#         else:
#             ax.fill_betweenx([y_start, y_end], ix, ix + 1, color='yellow', alpha=0.5)
    
#     if m <-99999999:
#             # if point1[1] > point2[1]:
#             #     # ax.fill_betweenx([y_start, y_end], ix-1, ix , color='yellow', alpha=0.5)
#             #     ax.fill_betweenx([y_end, y_start], ix-1, ix , color='yellow', alpha=0.5)
#             # if point1[1] < point2[1]:
#             ax.fill_betweenx([y_start, y_end], ix, ix + 1, color='yellow', alpha=0.5)
        
#     if m<  99999999:
#         # Check and store all y indices between y_start and y_end
#         y_indices = range(y_start, y_end) if y_at_x <= y_at_x_next else range(y_end, y_start,-1)
#         if m < 0:
#             y_indices = range(y_start, y_end) 
        
#         if m < -99999999:
#             y_indices = range(y_start, y_end, -1)
        
#     else:
        
#         y_indices = range(y_start, y_end) 
#     print("y_indices is : ",y_indices)
#     for iy in y_indices:
        
#         if m >99999999:
#                 highlighted_indices.append((ix, iy))
#         if m <-99999999:
#             # if point1[1] > point2[1]:
#             #     highlighted_indices.append((ix-1, iy))
#             # if point1[1] < point2[1]:
#             highlighted_indices.append((ix, iy))
#         if (m>0) and (m <99999999) :
#             if point1[0] < point2[0]:
#                 highlighted_indices.append((ix, iy))
#             if point1[0] > point2[0]:
#                 highlighted_indices.append((ix-1, iy-1))
#         if (m<0) and (m  > -99999999) :
#             # m<0
#             if point1[0] < point2[0]:
#                 highlighted_indices.append((ix, iy))
#             if point1[0] > point2[0]:
#                 highlighted_indices.append((ix-1, iy))
            
# # Set axis limits to show integer grid lines more clearly
# ax.set_xticks(np.arange(min(point1[0], point2[0]), max(point1[0], point2[0])+1, 1))
# ax.set_yticks(np.arange(min(point1[1], point2[1]), max(point1[1], point2[1])+1, 1))

# # Set labels and show plot
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_title('Plot of Points (0,0) and (4,6) with Line Connection')

# # Filter highlighted_indices
# if (point1[0], point1[1] ) in highlighted_indices:
#     highlighted_indices.remove((point1[0], point1[1] ) )
#     print("Element removed:", point1)
# if (point2[0], point2[1] ) in highlighted_indices:
#     highlighted_indices.remove((point2[0], point2[1] ) )
#     print("Element removed:", point2)

# # Print the highlighted grid cell indices
# print("Highlighted grid cell indices:", len(highlighted_indices))
# for index in highlighted_indices:
#     print(index)
    
# plt.show()





################3d line V1

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# def supercover_line(x0, y0, z0, x1, y1, z1):
#     """Calculate 3D supercover line coordinates and return the voxels the line passes through."""
#     x0, y0, z0 = int(np.floor(x0)), int(np.floor(y0)), int(np.floor(z0))
#     x1, y1, z1 = int(np.floor(x1)), int(np.floor(y1)), int(np.floor(z1))
#     points = []
#     dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
#     sx = -1 if x0 > x1 else 1
#     sy = -1 if y0 > y1 else 1
#     sz = -1 if z0 > z1 else 1
#     x, y, z = x0, y0, z0
#     err1, err2 = 0, 0
#     dx2 = dx * 2
#     dy2 = dy * 2
#     dz2 = dz * 2

#     if (dx >= dy) and (dx >= dz):
#         err1 = dy2 - dx
#         err2 = dz2 - dx
#         while x != x1:
#             points.append((x, y, z))
#             if err1 > 0:
#                 y += sy
#                 err1 -= dx2
#             if err2 > 0:
#                 z += sz
#                 err2 -= dx2
#             err1 += dy2
#             err2 += dz2
#             x += sx
#     elif (dy >= dx) and (dy >= dz):
#         err1 = dx2 - dy
#         err2 = dz2 - dy
#         while y != y1:
#             points.append((x, y, z))
#             if err1 > 0:
#                 x += sx
#                 err1 -= dy2
#             if err2 > 0:
#                 z += sz
#                 err2 -= dy2
#             err1 += dx2
#             err2 += dz2
#             y += sy
#     else:
#         err1 = dy2 - dz
#         err2 = dx2 - dz
#         while z != z1:
#             points.append((x, y, z))
#             if err1 > 0:
#                 y += sy
#                 err1 -= dz2
#             if err2 > 0:
#                 x += sx
#                 err2 -= dz2
#             err1 += dy2
#             err2 += dx2
#             z += sz
#     points.append((x1, y1, z1))
#     return points

# # Define points
# point1 = np.array([0, 0, 0])  # Starting point
# point2 = np.array([4, 6, 4])  # Ending point

# print("from pt", point1, "to", point2)

# # Create a 3D figure and axes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Compute the voxels through which the line passes
# voxels = supercover_line(point1[0], point1[1], point1[2], point2[0], point2[1], point2[2])

# # Mark voxels through which the line passes
# for voxel in voxels:
#     ax.bar3d(voxel[0], voxel[1], voxel[2], 1, 1, 1, color='yellow', alpha=0.1)

# # Plot the given points
# ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='red', s=100)

# # Connect the points with a line
# ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='blue', linewidth=2)

# # Setting grid
# ax.grid(True)

# # Setting labels
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# # Set title
# ax.set_title('3D Line Plot and Intersected Voxels')

# # Filter highlighted_indices
# if (point1[0], point1[1], point1[2] ) in voxels:
#     voxels.remove((point1[0], point1[1], point1[2]))
#     print("Element removed:", point1)
# if (point2[0], point2[1], point2[2] ) in voxels:
#     voxels.remove((point2[0], point2[1], point2[2]  ) )
#     print("Element removed:", point2)

# # Print the indices of the voxels intersected by the line
# print("Voxel indices intersected by the line:", len(voxels))
# for voxel in voxels:
#     print(voxel)

# # Show plot
# plt.show()



################3d line V2

