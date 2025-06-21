import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Define the RRT class
class RRT:
    def __init__(self, start, goal, map_limits, step_size, max_iters):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.map_limits = map_limits
        self.step_size = step_size
        self.max_iters = max_iters
        self.tree = [self.start]  # Initialize the tree with the start point
        self.parent = {0: None}  # Store parent of each node for path generation
        self.reachable = False

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def sample(self):
        # Randomly sample a point in the map limits
        return np.random.uniform(self.map_limits[0], self.map_limits[1], 2)

    def nearest_node(self, point):
        # Use KDTree to find the nearest point in the tree to the sampled point
        tree_points = np.array(self.tree)
        kdtree = KDTree(tree_points)
        nearest_index = kdtree.query(point)[1]
        return nearest_index

    def extend(self, nearest, new_point):
        distance = self.distance(nearest, new_point)
        direction = (new_point - nearest) / distance
        return nearest + direction * self.step_size
        #>>> return 
    
    def run(self):
        for i in range(self.max_iters):
            sampled_point = self.sample()
            nearest_index = self.nearest_node(sampled_point)
            nearest_point = self.tree[nearest_index]
            new_point = self.extend(nearest_point, sampled_point)

            # Add new point to the tree and record its parent
            self.tree.append(new_point)
            self.parent[len(self.tree) - 1] = nearest_index

            # Check if the goal is reached
            if self.distance(new_point, self.goal) < self.step_size:
                self.tree.append(self.goal)
                self.parent[len(self.tree) - 1] = len(self.tree) - 2
                print("Goal reached!")
                self.reachable = True
                break
        return self.tree, self.parent

    def get_path(self):
        # Reconstruct the path from goal to start using the parent dictionary
        path = [self.goal]
        current = len(self.tree) - 1
        while current is not None:
            path.append(self.tree[current])
            current = self.parent[current]
        return path[::-1]
    
    def get_new_point(self):
        path = self.get_path()
        return path[-2]

# Visualization of the RRT Algorithm
def visualize_rrt(rrt):
    tree, _ = rrt.run()

    fig, ax = plt.subplots()
    ax.set_xlim(rrt.map_limits[0][0], rrt.map_limits[1][0])
    ax.set_ylim(rrt.map_limits[0][1], rrt.map_limits[1][1])

    # Plot the tree
    for i in range(1, len(tree)):
        point = tree[i]
        parent_index = rrt.parent[i]
        parent_point = tree[parent_index]
        ax.plot([parent_point[0], point[0]], [parent_point[1], point[1]], 'b-')

    # Plot start and goal
    ax.plot(rrt.start[0], rrt.start[1], 'go', label="Start")
    ax.plot(rrt.goal[0], rrt.goal[1], 'ro', label="Goal")

    # Plot the final path
    path = rrt.get_path()
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'r--', linewidth=2, label="Path")

    ax.legend()
    plt.show()

# Set up the RRT parameters
start = [0, 0.45]
goal = [0.2, -0.36]
map_limits = [[-0.6, -0.6], [0.6,0.6]]
step_size = 0.1
# step_size = 5
max_nodes = 1000
max_nodes = 10

# Create RRT instance and visualize
rrt = RRT(start, goal, map_limits, step_size, max_nodes)
tree, parent = rrt.run()
# print(tree)
# print(parent)
path = rrt.get_path()
path = np.array(path)
print(path)
# print(type(path))
# print(path[-2])
print(rrt.get_new_point())
visualize_rrt(rrt)
