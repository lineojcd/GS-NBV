import torch

# Assuming the tensor is already on CPU
tensor_cpu = torch.tensor([[[-9.2245e-03, -1.5374e-02,  4.0600e-01],
                            [-8.3049e-03, -1.5298e-02,  4.0400e-01],
                            [-7.3939e-03, -1.5223e-02,  4.0200e-01],
                            [-6.5078e-03, -1.5185e-02,  4.0100e-01],
                            [-4.2758e-03, -1.1512e-02,  3.0400e-01],
                            [-3.6179e-03, -1.1512e-02,  3.0400e-01],
                            [-2.9504e-03, -1.0818e-02,  3.0300e-01],
                            [-2.2872e-03, -1.0129e-02,  3.0200e-01],
                            [-1.6283e-03, -1.0095e-02,  3.0100e-01],
                            [-9.7696e-04, -9.4441e-03,  3.0100e-01],
                            [-3.2565e-04, -8.1415e-03,  3.0100e-01],
                            [ 3.2458e-04, -7.4653e-03,  3.0000e-01],
                            [ 9.7050e-04, -7.4404e-03,  2.9900e-01],
                            [ 1.6121e-03, -6.7707e-03,  2.9800e-01],
                            [ 2.2569e-03, -5.4810e-03,  2.9800e-01],
                            [ 2.8920e-03, -4.8200e-03,  2.9700e-01],
                            [ 3.5228e-03, -4.8037e-03,  2.9600e-01],
                            [ 4.1492e-03, -4.7875e-03,  2.9500e-01],
                            [ 4.7875e-03, -4.7875e-03,  2.9500e-01],
                            [ 5.4075e-03, -4.7713e-03,  2.9400e-01],
                            [ 6.0231e-03, -4.1211e-03,  2.9300e-01],
                            [ 6.6571e-03, -4.1211e-03,  2.9300e-01],
                            [ 7.2662e-03, -3.4751e-03,  2.9200e-01],
                            [ 7.8981e-03, -2.8433e-03,  2.9200e-01],
                            [ 8.5299e-03, -2.2114e-03,  2.9200e-01],
                            [ 9.1304e-03, -2.2039e-03,  2.9100e-01],
                            [ 9.7266e-03, -2.1963e-03,  2.9000e-01],
                            [ 1.0354e-02, -2.1963e-03,  2.9000e-01],
                            [ 1.0944e-02, -1.5634e-03,  2.8900e-01],
                            [ 1.1569e-02, -9.3801e-04,  2.8900e-01],
                            [-1.7910e-02,  1.6755e-02,  5.3400e-01],
                            [-1.7843e-02,  1.7843e-02,  5.3200e-01],
                            [-1.6629e-02,  1.8923e-02,  5.3000e-01],
                            [-1.1792e-02,  2.3022e-02,  5.1900e-01],
                            [-1.0628e-02,  2.4052e-02,  5.1700e-01]]])


# Step 1: Calculate the mean position
mean_position = tensor_cpu.mean(dim=1)

# Step 2: Calculate the Euclidean distances of each point from the mean
distances = torch.norm(tensor_cpu - mean_position.unsqueeze(1), dim=2)

# Step 3: Calculate the standard deviation of these distances
std_dev = distances.std()

# Step 4: Filter out points within 2 standard deviations
mask = distances <= 2 * std_dev
filtered_points = tensor_cpu[mask].reshape(-1, 3)

# Step 5: Recalculate the mean using the filtered points
filtered_mean = filtered_points.mean(dim=0)

print("Filtered Mean:", filtered_mean)
print("Unfiltered Mean:", tensor_cpu.mean(dim=1))


# # Calculate the mean and standard deviation across all points
# mean = tensor_cpu.mean(dim=1)
# std = tensor_cpu.std(dim=1)
# print("mean and std are:", mean, std)
# print("tensor_cpu shape:",tensor_cpu.shape)
# print("mean shape:",mean.shape)

# # Calculate the Euclidean distances of each point from the mean
# distances = torch.norm(tensor_cpu - mean.unsqueeze(1), dim=2)
# print("distances", distances)
# print("distances shape:",distances.shape)

# # Filter out points within 2 standard deviations
# mask = distances <= 2 * std
# print("mask", mask)
# print("mask shape:",mask.shape)

# filtered_points = tensor_cpu[mask.expand_as(tensor_cpu)].reshape(-1, 3)

# # Recalculate the mean using the filtered points
# filtered_mean = filtered_points.mean(dim=0)

# print("Filtered Mean:", filtered_mean)
