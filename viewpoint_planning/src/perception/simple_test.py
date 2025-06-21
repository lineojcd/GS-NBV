import time
if False:
    # List of tuples
    data = [(0.0, 0.96957535), (1.0, 0.5026132), (1.0, 0.4217139)]

    # Convert to a dictionary with lists as values
    result_dict = {}
    for key, value in data:
        if key not in result_dict:
            result_dict[key] = value
        else:
            if value > result_dict[key]:
                result_dict[key] = value

    # Display the resulting dictionary
    print(result_dict)



# ##########################


import torch
if False:
    # Data and masks
    data = [(0.0, 0.96957535), (1.0, 0.3026132), (1.0, 0.4217139)]
    masks = torch.rand(3, 640, 640)  # Assuming random masks for demonstration, replace with your actual mask data

    # Convert list to dictionary and fetch the corresponding masks
    result_dict = {}
    result_masks = []

    for i, (key, value) in enumerate(data):
        if key not in result_dict:
            result_dict[key] = value
            result_masks.append(masks[i])

    # Convert result_masks list to a torch tensor if needed
    result_masks = torch.stack(result_masks)

    # Display the results
    print("Filtered Dictionary:", result_dict)
    print("Related Masks Shape:", result_masks.shape)

    # Optional: To display the actual mask data, uncomment the following line
    # print("Related Masks Data:", result_masks)


# Initialize the data
semantic_res = [(0.0, 0.96957535), (1.0, 0.35026132), (1.0, 0.4217139)]
# masks = torch.rand(3, 2, 2)  # Example masks, replace with your actual masks
masks = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])

print(masks)
# Generate a list of tuples with key-value pairs and mask shapes
# result = [(key, masks[idx].unsqueeze(0).shape) for idx, (key, value) in enumerate(semantic_res)]
result_list = []
for i in range(len(semantic_res)):
    print("semantic_res[i][0]",semantic_res[i][0])
    result_list.append((semantic_res[i][0], masks[i,:,:]))
    
    
# Display the result
print(semantic_res)
# print(result)


    # Convert to a dictionary with lists as values
result_dict = {}
mask_dict = {}
for idx, (key, value) in enumerate(semantic_res):
    if key not in result_dict:
        result_dict[key] = value
        mask_dict[key] = result_list[idx][1]
    else:
        if value > result_dict[key]:
            result_dict[key] = value
            mask_dict[key] = result_list[idx][1]

# Display the resulting dictionary
print(">>>>>>result_dict is")
result_dict[2.0] = result_dict[0.0]
print(result_dict)
print("mask_dict is")
print(mask_dict)
# for key, value in mask_dict.items():
#     print(f"Key: {key}, Size: {value}")
#     print(value)

for key, tensor in mask_dict.items():
    print(f"Key: {key}, Tensor:\n{tensor}\n")
# print(mask_dict[0])


# Extract the tensors from the dictionary and stack them
combined_mask = torch.stack(list(mask_dict.values()))
# Print the resulting combined mask
print("Combined Mask:")
print(combined_mask)
print("Shape of the Combined Mask:", combined_mask.shape)

# Check if the key 1 is in the dictionary
if 1 in result_dict:
    print("Key 1 is in the dictionary.")
else:
    print("Key 1 is not in the dictionary.")


tmpresult_dict = {}
print(len(tmpresult_dict))
# tmp_res_lst = list(tmpresult_dict.keys())
# print(type(tmp_res_lst.keys()))

m1_tensor = torch.tensor(
    [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
     [0., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
     [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]
    )
m2_tensor = torch.tensor(
    [[1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    )


m1_mask = masks[0, :, :]
m2_mask = masks[1, :, :]

print("Shape of the m1_mask Mask:", m1_mask.shape)
print("Shape of the m2_mask Mask:", m2_mask.shape)

# print(m1_tensor+ m2_tensor)
# Perform element-wise multiplication
overlap_tensor = m1_tensor * m2_tensor

# Check if there are any non-zero elements in the result
overlap_exists = overlap_tensor.nonzero().size(0) > 0

# Output whether there is overlap
print("Is there overlap?:", overlap_exists)

# Find the indices of non-zero elements (locations of overlaps)
overlap_indices = overlap_tensor.nonzero(as_tuple=False)

# Print the locations of the overlaps
print("Locations of overlap (row, column):")
print(overlap_indices)


# Define the depth_map tensor
depth_map = torch.tensor([
    [10,  1,  8,  6,  5,  2,  1,  4,  5,  1],
    [ 3,  9,  7,  6,  8,  4,  2,  3,  7,  2],
    [ 1,  9,  3, 10,  8,  1,  6,  4,  8,  7],
    [ 5,  1,  9,  6,  4,  5,  2,  9,  4, 10],
    [ 6,  1,  7,  6,  3,  8,  4, 10,  7,  5],
    [ 8,  9,  6,  1,  2,  4,  9,  7,  1,  6],
    [ 9,  7,  8,  1, 10,  5,  4,  8,  1,  6],
    [10,  2,  8,  9,  4,  5,  7,  9,  2,  6],
    [ 7,  5,  8,  6,  3,  4,  1,  9, 10,  8],
    [ 3,  7,  9,  5,  8,  2,  6,  9, 10,  1]
])

# Accessing the depth values using indices
depth_values = depth_map[overlap_indices[:, 0], overlap_indices[:, 1]]

# Printing the result
print("Depth values at the specified indices:")
print(depth_values)

# my_test_dict = {0.0: 0.96957535, 1.0:0.35026132, 1.0: 0.4217139}
# print(len(my_test_dict))
# for key, value in my_test_dict.items():
#     print(f"Key: {key}, Value: {value}")

    
    
# mymasks = torch.tensor([
# [[1, 2], [3, 4]],
# [[5, 6], [7, 8]],
# [[9, 10], [11, 12]]
# ])

# print(mymasks[0])



# # Example tensor with shape (2, 3, 4)
# semantics = torch.arange(24).view(2, 3, 4)
# print("Original semantics tensor:")
# print("Original semantics tensor shape:", semantics.shape)
# print(semantics)

# # Reshape semantics tensor to (-1, 2)
# reshaped_semantics = semantics.view(-1, 2)
# print("Reshaped semantics tensor:")
# print(reshaped_semantics)



testLst = [0,1, 0, 0, 10]
testSet= set(testLst)
# Print the resulting set
print("Set from the list:", testSet)

# Find the minimum value in the set
min_value = min(testSet)

# Find the maximum value in the set
max_value = max(testSet)

# Print the minimum and maximum values
print("Minimum value in the set:", min_value)
print("Maximum value in the set:", max_value)

# Calculate the sum of all elements in the set
total_sum = sum(testSet)

# Calculate the number of elements in the set
num_elements = len(testSet)

# Calculate the average
average = total_sum / num_elements

# Print the average
print("Average of the set:", average)


start_time = time.time()
print(f"extract_fruit_contour prediction time: {time.time() - start_time} seconds")