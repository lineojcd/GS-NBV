import torch
import numpy as np

def fit_3dline_by_PCA(tensor_pts):
    # Convert the tensor to a NumPy array
    points = tensor_pts.numpy().reshape(-1, 3)

    # Step 2: Center the data
    mean = np.mean(points, axis=0)
    centered_data = points - mean

    # Step 3: Compute the covariance matrix
    covariance_matrix = np.cov(centered_data.T)

    # Step 4: Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 5: Select the principal component (eigenvector with the largest eigenvalue)
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]


    # Step 6: Normalize the principal component
    normalized_principal_component = principal_component / np.linalg.norm(principal_component)

    # Output the normalized direction of the line
    print("Normalized direction of the best-fit line by PCA:", normalized_principal_component)
    return normalized_principal_component

def fit_3dline_by_2points(top_pt, bottom_pt):
    # Extract the points
    # point1 = fruit_wf[0, 0]  # First point
    # point2 = fruit_wf[0, 1]  # Second point
    
    # Compute the direction vector from point1 to point2
    direction_vector = top_pt - bottom_pt

    # Normalize the direction vector
    norm = torch.norm(direction_vector)
    normalized_direction_vector = direction_vector / norm

    # Output the normalized direction of the line
    print("Normalized direction of the best-fit line by top and bottom points:", normalized_direction_vector.numpy())
    return normalized_direction_vector.numpy()
    
if __name__ == '__main__':
    # Your tensor
    fruit_wf = torch.tensor([[[0.5111, -0.2568, 1.2027],
                          [0.4925, -0.2595, 1.1579]]])

    axis = fit_3dline_by_PCA(fruit_wf)
    print("axis is " , axis)
    
    axis = fit_3dline_by_2points(fruit_wf)
    print("axis is " , axis)

