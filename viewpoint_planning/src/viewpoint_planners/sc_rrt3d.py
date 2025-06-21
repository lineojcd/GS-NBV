import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import rospy

# Define the RRT class for 3D
class RRT3D:
    def __init__(self):
        self.max_nodes = rospy.get_param("SCNBV_RRT_max_node", "10") 
        self.step_size = rospy.get_param("SCNBV_RRT_step_size", "0.1") 
        self.vp_sampling_Rmin = rospy.get_param("SCNBV_sampling_Rmin", "0.21")
        
    def set_sampling_range(self,map_limits):
        self.map_limits = map_limits
        
    def set_obstacle_region(self,tree_obstacle_region, Q_est):
        # a region that is excluded from moveit
        self.obstacle_region = np.array(tree_obstacle_region)
        self.Q_est = Q_est

    def set_nodes(self,start,goal):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.tree = [self.start]  # Initialize the tree with the start point
        self.parent = {0: None}  # Store parent of each node for path generation
    
    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def sample(self):
        while True:
            point = np.random.uniform(self.map_limits[0], self.map_limits[1], 3)
            if not self.in_obstacle(point):
                return point

    def nearest_node(self, point):
        # Use KDTree to find the nearest point in the tree to the sampled point
        tree_points = np.array(self.tree)
        kdtree = KDTree(tree_points)
        nearest_index = kdtree.query(point)[1]
        return nearest_index

    def extend(self, nearest, new_point):
        direction = (new_point - nearest) / self.distance(nearest, new_point)
        new_point = nearest + direction * self.step_size
        if self.in_obstacle(new_point):
            return nearest
        return new_point
    
    def in_tree_obstacle(self, point):
        return np.all(point >= self.obstacle_region[0]) and np.all(point <= self.obstacle_region[1])
    
    def in_sphere_obstacle(self, point):
        return np.linalg.norm(point - self.Q_est) < self.vp_sampling_Rmin

    def in_obstacle(self, point):
        rospy.logwarn("Obstacle including tree obstacle and sphere obstacle")
        if self.in_tree_obstacle(point) and self.in_sphere_obstacle(point):
            return True
        else:
            return False
    
    def run(self):
        for i in range(self.max_nodes):
            sampled_point = self.sample()
            nearest_index = self.nearest_node(sampled_point)
            nearest_point = self.tree[nearest_index]
            new_point = self.extend(nearest_point, sampled_point)
            
            # Add new point to the tree and record its parent
            if not np.array_equal(new_point, self.tree[nearest_index]):
                self.tree.append(new_point)
                self.parent[len(self.tree) - 1] = nearest_index

            # Check if the goal is reached
            if self.distance(new_point, self.goal) < self.step_size:
                self.tree.append(self.goal)
                self.parent[len(self.tree) - 1] = len(self.tree) - 2
                print("Goal reached!")
                break
        return self.tree, self.parent

    def get_path(self):
        # Reconstruct the path from goal to start using the parent dictionary
        path = [self.goal]
        current = len(self.tree) - 1
        while current is not None:
            path.append(self.tree[current])
            current = self.parent[current]
        # return path[::-1]
        return path
    
    def get_new_point(self):
        path = self.get_path()
        return path[1]

# Visualization of the 3D RRT Algorithm
def visualize_rrt_3d(rrt):
    tree, _ = rrt.run()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(rrt.map_limits[0][0], rrt.map_limits[1][0])
    ax.set_ylim(rrt.map_limits[0][1], rrt.map_limits[1][1])
    ax.set_zlim(rrt.map_limits[0][2], rrt.map_limits[1][2])

    # Plot the tree
    for i in range(1, len(tree)):
        point = tree[i]
        parent_index = rrt.parent[i]
        parent_point = tree[parent_index]
        ax.plot([parent_point[0], point[0]], [parent_point[1], point[1]], [parent_point[2], point[2]], 'b-')

    # Plot start and goal
    ax.scatter(rrt.start[0], rrt.start[1], rrt.start[2], color='green', label="Start", s=100)
    ax.scatter(rrt.goal[0], rrt.goal[1], rrt.goal[2], color='red', label="Goal", s=100)

    # Plot the final path
    path = rrt.get_path()
    path = np.array(path)
    print("visualize_rrt_3d : ", path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r--', linewidth=2, label="Path")
    # Visualize obstacle
    obstacle_size = np.abs(rrt.obstacle_region[1] - rrt.obstacle_region[0])
    ax.bar3d(rrt.obstacle_region[0][0], rrt.obstacle_region[0][1], rrt.obstacle_region[0][2],
             obstacle_size[0], obstacle_size[1], obstacle_size[2],
             color='red', alpha=0.1, zsort='max')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()




if __name__ == '__main__':
    pass
    # Set up the RRT parameters for 3D
    start = [0, 0.35, 0.0]
    goal = [-0.21,-0.26, 0.0]
    map_limits = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    step_size = 0.1
    max_nodes = 20
    obstacle_region = [[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]]


    # Create RRT3D instance and visualize
    rrt_3d = RRT3D(start, goal, map_limits, step_size, max_nodes, obstacle_region)
    tree, parent = rrt_3d.run()
    path = rrt_3d.get_path()
    path = np.array(path)
    print("outside: ",path)
    print(rrt_3d.get_new_point())

    visualize_rrt_3d(rrt_3d)