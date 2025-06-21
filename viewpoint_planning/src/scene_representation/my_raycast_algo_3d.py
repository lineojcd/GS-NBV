import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch


def raycast_3d(point1_idx, point2_idx, center_offset = np.array([0.5, 0.5, 0.5]), verbose = False):
    # Test group: up right
    pt1 = point1_idx + center_offset
    pt2 = point2_idx + center_offset
    
    direction = pt2 - pt1
    step_size = 0.3  # Small step size to ensure we catch all voxels
    num_steps = int(np.linalg.norm(direction) / step_size)
    steps = np.linspace(0, 1, num_steps)
    # List to store highlighted voxel indices
    highlighted_voxels = []
    
    # Loop through steps to find all voxels the line passes through
    for st in steps:
        current_point = pt1 + st * direction
        voxel = tuple(np.floor(current_point).astype(int))
        if voxel not in highlighted_voxels:
            highlighted_voxels.append(voxel)
            
    # Filter highlighted_indices
    if (point1_idx[0], point1_idx[1], point1_idx[2] ) in highlighted_voxels:
        highlighted_voxels.remove((point1_idx[0], point1_idx[1], point1_idx[2]))
        if verbose:
            print("Element removed:", point1_idx)
    if (point2_idx[0], point2_idx[1], point2_idx[2] ) in highlighted_voxels:
        highlighted_voxels.remove((point2_idx[0], point2_idx[1], point2_idx[2]  ) )
        if verbose:
            print("Element removed:", point2_idx)
    
    if verbose:
        # Print the highlighted voxel indices
        print("Highlighted voxel indices:", len(highlighted_voxels))
        for voxel in highlighted_voxels:
            print(voxel)
    
    return highlighted_voxels

def check_visibility(nparray_2d, traversed_voxels_lst_of_tuple3, threshold = 0 ):
    # Convert highlighted_voxels to a set of tuples
    highlighted_voxels_set = set(traversed_voxels_lst_of_tuple3)

    # Check if elements in semantic_idx are in highlighted_voxels
    matches = [tuple(coord) in highlighted_voxels_set for coord in nparray_2d]
    visible_cnt = sum(matches)
    # print("visible_cnt is :",visible_cnt)
    if visible_cnt == threshold:
        return True
    else:
        return False

def check_roi_visibility(occupied_indices, traversed_voxels_lst_of_tuple3, threshold = 0 ):
    # Check if elements in traversed_voxels are in occupied_indices
    matches = [coord in occupied_indices for coord in traversed_voxels_lst_of_tuple3]
    occupied_cnt = sum(matches)
    # print("visible_cnt is :",visible_cnt)
    if visible_cnt == threshold:
        return True
    else:
        return False

def check_roi_visibility_np(occupied_indices, traversed_voxels_lst_of_tuple3, threshold = 0 ):
    # Convert occupied_indices to a set of tuples for fast lookup
    occupied_indices_set = {tuple(coord.cpu().numpy()) for coord in occupied_indices}
    
    # Check if elements in traversed_voxels are in occupied_indices_set
    matches = [coord in occupied_indices_set for coord in traversed_voxels_lst_of_tuple3]
    occupied_cnt = sum(matches)
    
    # Check the threshold
    if occupied_cnt == threshold:
        return True
    else:
        return False

def check_roi_visibility_tensor(occupied_indices, traversed_voxels_lst_of_tuple3, threshold = 0 ):
    # Convert traversed_voxels_lst_of_tuple3 to a tensor
    traversed_voxels_tensor = torch.tensor(traversed_voxels_lst_of_tuple3, dtype=torch.float32, device=occupied_indices.device)
    # traversed_voxels_tensor = torch.tensor(traversed_voxels_lst_of_tuple3, dtype=torch.float32, device='cuda:0')
    # occupied_indices = torch.tensor(occupied_indices, dtype=torch.float32, device='cuda:0')
    
    # Compare traversed_voxels_tensor with occupied_indices
    matches = (traversed_voxels_tensor[:, None] == occupied_indices).all(dim=-1).any(dim=-1)
    occupied_cnt = matches.sum().item()  # Get count of matches
    
    # Check the threshold
    if occupied_cnt == threshold:
        return True
    else:
        return False
    
def check_semantics_visibility_tensor(occupied_indices, traversed_voxels_lst_of_tuple3, threshold = 0 ):
    return check_roi_visibility_tensor(occupied_indices, traversed_voxels_lst_of_tuple3, threshold)

def visulize(start_pt, end_pt, highlighted_voxels, center_offset = np.array([0.5, 0.5, 0.5])):
    # Test group: up right
    point1 = start_pt + center_offset
    point2 = end_pt + center_offset
    
    # Create the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the given points
    ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='red')
    # Connect the points with a line
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='blue')

    # Set grid for visualization
    ax.grid(True)
    
    # Draw the voxels the line passes through
    for voxel in highlighted_voxels:
        x, y, z = voxel
        voxel_corners = np.array([[x, y, z],
                                [x+1, y, z],
                                [x+1, y+1, z],
                                [x, y+1, z],
                                [x, y, z+1],
                                [x+1, y, z+1],
                                [x+1, y+1, z+1],
                                [x, y+1, z+1]])
        faces = [[voxel_corners[j] for j in [0, 1, 2, 3]], # bottom
                [voxel_corners[j] for j in [4, 5, 6, 7]], # top
                [voxel_corners[j] for j in [0, 1, 5, 4]], # front
                [voxel_corners[j] for j in [2, 3, 7, 6]], # back
                [voxel_corners[j] for j in [1, 2, 6, 5]], # right
                [voxel_corners[j] for j in [4, 7, 3, 0]]] # left
        ax.add_collection3d(Poly3DCollection(faces, color='lightpink', alpha=0.2))
        
    # Set labels and show plot
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of Points and Connecting Line with Highlighted Voxels')
    
    plt.show()
    
    
if __name__ == "__main__": 
    
    
    # Example data
    semantic_idx = np.array([[82, 104, 120],
                            [82, 104, 122],
                            [83, 103, 118],
                            [94, 107, 122],
                            [95, 103, 118],
                            [95, 103, 119],
                            [95, 103, 123],
                            [95, 103, 124],
                            [95, 103, 125],
                            [95, 104, 122],
                            [57, 87, 123],
                            [95, 105, 124]])

    highlighted_voxels = [(14, 59, 128), (14, 60, 128), (15, 60, 128), (16, 61, 128), (17, 61, 128), 
                        (17, 62, 128), (18, 62, 127), (19, 63, 127), (20, 63, 127), (20, 64, 127), 
                        (21, 64, 127), (22, 65, 127), (23, 65, 127), (23, 66, 127), (24, 66, 127), 
                        (25, 67, 127), (26, 67, 127), (26, 68, 126), (27, 68, 126), (28, 69, 126), 
                        (29, 69, 126), (29, 70, 126), (52, 85, 123), (53, 85, 123), (54, 85, 123), 
                        (54, 86, 123), (55, 86, 123), (55, 87, 123), (56, 87, 123), (57, 87, 123)]
    
    # Convert highlighted_voxels to a set of tuples
    highlighted_voxels_set = set(highlighted_voxels)

    # Check if elements in semantic_idx are in highlighted_voxels
    matches = [tuple(coord) in highlighted_voxels_set for coord in semantic_idx]

    # Print the result
    print("Matches:", matches)
    print("Matched Indices:", [i for i, match in enumerate(matches) if match])
    
    res = check_visibility(semantic_idx, highlighted_voxels)
    print(res)
    
    point1_idx = np.array([0, 0, 0])  # Starting point
    point2_idx = np.array([40, 60, 40])  # Ending point
    highlighted_voxels = raycast_3d(point1_idx, point2_idx)
    print("highlighted_voxels = ", highlighted_voxels)
    visulize(point1_idx, point2_idx, highlighted_voxels)
    
    



