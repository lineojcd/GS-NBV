import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_auxiliary_vector(axis):
    # Create an orthogonal vector to axis
    if (axis == np.array([1, 0, 0])).all():
        first_vector = np.cross(axis, np.array([0, 1, 0]))
    else:
        first_vector = np.cross(axis, np.array([1, 0, 0]))
    
    first_vector = first_vector / np.linalg.norm(first_vector)
    second_vector = np.cross(axis, first_vector)
    second_vector = -1* second_vector / np.linalg.norm(second_vector) +np.array([0, 0, 0])
    # return orthogonal_vector, second_vector
    
    # TODO: need to see the relation of this vectors
    print("orthogonal_vector and  second_vector are", first_vector, second_vector)
    return first_vector, second_vector


def point_Circle_plane_projection(point, center, axis):
    return point - np.dot(point - center, axis) * axis

# TODO: delete this function when finished
def get_line_circle_intersections( radius, line_pta, line_ptb):
    # Calculate the direction vector of the line
    dir = line_ptb -line_pta
    # Circle's equation in the XY plane: (x - h)^2 + (y - k)^2 = r^2
    # Here h = 0, k = 0, r = 3 (center at origin and radius 3)

    # Parametric line equations: x = x0 + t*dx, y = y0 + t*dy
    # x0, y0 are coordinates of tree_a; dx, dy are components of the direction vector
    x0, y0 = line_pta[:2]
    dx, dy = dir[:2]
    # Substitute the line's parametric equations into the circle's equation and solve for t
    # (x0 + t*dx)^2 + (y0 + t*dy)^2 = radius^2
    # Expanding and rearranging to form a quadratic equation At^2 + Bt + C = 0
    A = dx**2 + dy**2
    B = 2 * (x0 * dx + y0 * dy)
    C = x0**2 + y0**2 - radius**2
    
    # Solving the quadratic equation using the quadratic formula
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        print("No intersection")
    else:
        t1 = (-B - np.sqrt(discriminant)) / (2 * A)
        t2 = (-B + np.sqrt(discriminant)) / (2 * A)
        
        # Calculate the intersection points
        intersect_a = line_pta + t1 * dir
        intersect_b = line_pta + t2 * dir
        
        print("Workspace intersection Point 1:", intersect_a)
        print("Workspace intersection Point 2:", intersect_b)
    return intersect_a, intersect_b

def find_line_circle_intersect(P1,P2, center,radius):
    # Direction vector of the line
    direction = P2 - P1

    # Parameterized line equation r(t) = P1 + t * direction
    # We need to find t such that |r(t) - center| = radius

    # Expand the equation |r(t) - center|^2 = radius^2
    # (P1 + t*direction - center) . (P1 + t*direction - center) = radius^2
    # Simplify to At^2 + Bt + C = 0 form
    A = np.dot(direction, direction)
    B = 2 * np.dot(direction, P1 - center)
    C = np.dot(P1 - center, P1 - center) - radius**2

    # Solve quadratic equation At^2 + Bt + C = 0
    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        print("No intersection")
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-B + sqrt_discriminant) / (2 * A)
        t2 = (-B - sqrt_discriminant) / (2 * A)
        
        # Calculate intersection points
        intersection_point_1 = P1 + t1 * direction
        intersection_point_2 = P1 + t2 * direction
        
        print("Intersection Point 1:", intersection_point_1)
        print("Intersection Point 2:", intersection_point_2)

    return intersection_point_1, intersection_point_2


def get_void_angles(obstacle_points_projection, center):
    intersection_angles = []
    for pt in obstacle_points_projection:
        direction = pt - center
        # angle = np.arctan2(direction[1], direction[0]) - np.pi/2
        angle = np.arctan2(direction[1], direction[0]) 
        if angle < 0:
            angle  =  2* np.pi + angle
        intersection_angles.append(angle)
    
    min_a = min(intersection_angles)
    max_a = max(intersection_angles)
    return min_a,max_a

def get_void_angles_update(query_pt, center, first_pt):
    v  = query_pt - center
    u  = first_pt - center
    # Calculate the dot product of vectors v and u
    dot_product = np.dot(v, u)
    
    # Calculate the norms (magnitudes) of each vector
    norm_v = np.linalg.norm(v)
    norm_u = np.linalg.norm(u)

    # Compute the cosine of the angle using the dot product formula
    cos_theta = dot_product / (norm_v * norm_u)
    
    # Compute the angle in radians
    angle_radians = np.arccos(cos_theta)
    
    print("query_pt and first_pt:" ,query_pt, first_pt)
    if query_pt[1]<first_pt[1]:
        print("In")
        angle_radians += np.pi
        
    # Optionally convert the angle to degrees
    if False:
        angle_degrees = np.degrees(angle_radians)
    return angle_radians


def estimate_fruit_geo_info():
    return np.array([0, 0, 0]), np.array([0, 0, 1]), 3

def check_points_on_cicle(pointa, center, radius = 3):
    # Calculate the distance using numpy's linear algebra norm function
    distance = np.linalg.norm(pointa - center)
    # print("abs(distance - radius) is ",abs(distance - radius) )
    print("distance ", distance  )
    on_circle = abs(distance - radius) <= 0.01
    return on_circle

# Draw a circle
def draw_circle(cam_position, center, obstacle_points,axis, radius, max_angle, nb_vp_samples = 4, plot = True, samples=360, auxiliary_plane = True):
    # Normalize the axis vector
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    # Create auxiliary vectors to axis
    first_vector, second_vector = get_auxiliary_vector(axis)
    print("orthogonal_vector and  second_vector are", first_vector, second_vector)
    # Convert max_angle from degrees to radians
    max_angle_rad = np.radians(max_angle)

    # Generate points on the circle
    # Half-circle from 0 to 180 degrees
    # theta = np.linspace(0, 2 * np.pi, num_samples)
    # theta_bound = np.linspace(0, 2 * np.pi, nb_vp_samples, endpoint=False)  # Ensures points are evenly spaced
    # theta = np.linspace(0, 2 * np.pi, nb_vp_samples, endpoint=False)  # Ensures points are evenly spaced
    theta = np.linspace(-np.pi/2, 3*np.pi/2, nb_vp_samples, endpoint=False)  # Ensures points are evenly spaced
    # theta = np.linspace(0, np.pi/2, nb_vp_samples, endpoint=False)  # Ensures points are evenly spaced
    theta = np.linspace(0, 2* np.pi, samples, endpoint=True)  # Ensures points are evenly spaced
    # vp_points = np.array([center + radius * (np.cos(t) * orthogonal_vector + np.sin(t) * second_vector) for t in theta])
    vp_points = np.array([center + radius * (np.cos(t) * second_vector + np.sin(t) * first_vector) for t in theta])
    print("vp_points[0] is ",vp_points[0])
    # theta_bound = np.linspace(0, max_angle_rad,2)  # Ensures points are evenly spaced
    # theta_bound = np.linspace( -np.pi/2, np.pi/2, 2)
    # vp_points_bound = np.array([center + radius * (np.cos(t) * orthogonal_vector + np.sin(t) * second_vector) for t in theta_bound])
    # print("vp_points_bound is ",vp_points_bound)
    print("len of theta and vp_points:", len(theta) ,len(vp_points) )
    
    tree_center =  np.array([-1, -1, 1])
    tree_center_aux =  np.array([0, -1, 1])
    # Calculate the projection of the tree_center onto the plane
    tree_projection = point_Circle_plane_projection(tree_center, center, axis)
    tree_aux_projection = point_Circle_plane_projection(tree_center_aux, center, axis)
    # tree_p_b = tree_projection + second_vector
    print("tree_projection and tree_aux_projection are ",tree_projection, tree_aux_projection)
    # intersect_a, intersect_b = get_line_circle_intersections(radius, tree_projection, second_vector)
    # intersect_a, intersect_b = get_line_circle_intersections(radius, tree_projection, tree_aux_projection)
    intersect_a, intersect_b = find_line_circle_intersect( tree_projection, tree_aux_projection, center,radius)
    # check intersect_a and intersect_b
    resa = check_points_on_cicle(intersect_a, center)
    resb = check_points_on_cicle(intersect_b, center)
    print("intersect_a on circle:", resa)
    print("intersect_b on circle:", resb)
    
    # Conduct obstacle_points project
    obstacle_points_projection = []
    for pt in obstacle_points:
        v = point_Circle_plane_projection(pt, center, axis)
        obstacle_points_projection.append(v)
    obstacle_points_projection = np.array(obstacle_points_projection)
    print("obstacle_points are: ",obstacle_points)
    print("obstacle_points_projection are: ",obstacle_points_projection)
    
    # Plot lines from extra points to center and their intersection points
    Obs_intersecting = []
    # print('Obs_intersecting', Obs_intersecting)
    for pt in obstacle_points_projection:
        # Compute intersection points
        # Normalize point vector
        dir = pt - center
        v = center + dir / np.linalg.norm(dir) * radius
        # v = center + pt / np.linalg.norm(pt) * radius
        print("Obstacle intersecting points coords: ", v)
        # Obs_intersecting[i] = intersecting_point
        Obs_intersecting.append(v)
        # ax.scatter(*pt, color='Silver', s=40,label='Obstacle Points Projection')
    
    # obs_int_min_angle, obs_int_max_angle =  get_void_angles(Obs_intersecting, center)
    # print("obs_int_min_angle and  obs_int_max_angle are: ", obs_int_min_angle, obs_int_max_angle)
    
    Obs_intersecting = np.array(Obs_intersecting)
    
    # # Calculate intersection angles and find range
    # intersection_angles = []
    # for pt in obstacle_points_projection:
    #     direction = pt - center
    #     angle = np.arctan2(direction[1], direction[0]) - np.pi/2
    #     intersection_angles.append(angle)
    
    # min_angle = min(intersection_angles)
    # max_angle = max(intersection_angles)
    
    # Calculate intersection angles and find range
    min_angle, max_angle =  get_void_angles(obstacle_points_projection, center)
    print("old function:min_angle and  max_angle are: ", min_angle, max_angle)
    
    first_pt = vp_points[0]
    min_angle = get_void_angles_update(Obs_intersecting[0], center, first_pt)
    max_angle = get_void_angles_update(Obs_intersecting[-1], center, first_pt)
    print("min_angle and  max_angle are: ", min_angle, max_angle)
    
    # Determine which points are within the intersection angle range
    
    in_range = (theta >= min_angle) & (theta <= max_angle)
    # print("in_range are: ", in_range)
    # print("thetas are: ", theta)
    first_true_index = np.where(in_range)[0][0]
    print("first_true_index are: ", first_true_index)
    print("related theta is :", theta[first_true_index-1])
    print( "vp_points[26,:][0]", vp_points[26,:])
    print( "vp_points[27,:][-1]", vp_points[27,:])
    print( "vp_points[28,:][-1]", vp_points[28,:])
    
    
    # Calculate intersection angles and find range
    workspace_intersection_pt = np.array([intersect_a, intersect_b])
    workspace_min_angle, workspace_max_angle =  get_void_angles(workspace_intersection_pt, center)
    print("old function: workspace_min_angle and  workspace_max_angle are: ", workspace_min_angle, workspace_max_angle)
    
    workspace_min_angle =  get_void_angles_update(workspace_intersection_pt[0], center, first_pt)
    workspace_max_angle =  get_void_angles_update(workspace_intersection_pt[-1], center, first_pt)
    print("workspace_min_angle and  workspace_max_angle are: ", workspace_min_angle, workspace_max_angle)
    # Determine which points are within the intersection angle range
    workspace_in_range = (theta >= workspace_min_angle) & (theta <= workspace_max_angle)
    # print("workspace_in_range is ",workspace_in_range)
    
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plotting the points and the center with the axis
     # Color points accordingly, exclude the first and last points from the "in range" classification
     
    print("theta[first_true_index-10]: ",theta[first_true_index-10])
    print( "First point is ",vp_points[0])
    ax.scatter(vp_points[0, 0], vp_points[0 , 1], vp_points[0 , 2], s=100,color='red', label='First Pose')
    # ax.scatter(vp_points[first_true_index-10, 0], vp_points[first_true_index-10 , 1], vp_points[first_true_index-10 , 2], s=100,color='Green', label='First before In range Pose')
    # ax.scatter(vp_points[first_true_index, 0], vp_points[first_true_index , 1], vp_points[first_true_index , 2], s=100,color='red', label='First In range Pose')
    ax.scatter(vp_points[in_range, 0], vp_points[in_range , 1], vp_points[in_range , 2], color='grey', label='Void Pose')
    ax.scatter(vp_points[workspace_in_range, 0], vp_points[workspace_in_range , 1], vp_points[workspace_in_range , 2], color='Navy', label='Unreachable Pose')
    # ax.scatter(vp_points[~in_range, 0], vp_points[~in_range, 1], vp_points[~in_range, 2], color='magenta', label='Valid Points')
    ax.scatter(vp_points[~in_range & ~workspace_in_range, 0], vp_points[~in_range & ~workspace_in_range, 1], vp_points[~in_range & ~workspace_in_range, 2], color='magenta', label='Valid Pose')
    # ax.scatter(vp_points[workspace_in_range & (theta > theta[0]) & (theta < theta[-1]), 0], vp_points[workspace_in_range & (theta > theta[0]) & (theta < theta[-1]), 1], vp_points[workspace_in_range & (theta > theta[0]) & (theta < theta[-1]), 2], color='grey', label='Unreachable Pose')
    # ax.scatter(vp_points[~workspace_in_range, 0], vp_points[~workspace_in_range, 1], vp_points[~workspace_in_range, 2], color='red', label='Unreachable Pose')
    # ax.scatter(vp_points[workspace_in_range & (theta > theta[0]) & (theta < theta[-1]), 0], vp_points[workspace_in_range & (theta > theta[0]) & (theta < theta[-1]), 1], vp_points[workspace_in_range & (theta > theta[0]) & (theta < theta[-1]), 2], color='grey', label='Unreachable Pose')
    
    # ax.scatter(vp_points_bound[:,0], vp_points_bound[:,1], vp_points_bound[:,2], color='cyan',s=40,  label='Bound Points')
    ax.scatter(*intersect_a, color='cyan',s=40,  label='Workspace Bound Points')
    ax.scatter(*intersect_b, color='cyan',s=40)
    # ax.scatter(vp_points_bound[:,0], vp_points_bound[:,1], vp_points_bound[:,2], color='cyan',s=40,  label='Workspace Bound Points')
    ax.scatter(*center, color='red', s=80, label='Center')
    # ax.scatter(obstacle_points[:,0],obstacle_points[:,1],obstacle_points[:,2], color='Black', s=40, label='Obstacle Points')
    ax.scatter(*obstacle_points[0], color='Black', s=40, label='Obstacle Points')
    ax.scatter(*obstacle_points[-1], color='Black', s=40)
    ax.scatter(*obstacle_points_projection[0], color='Silver', s=40,label='Obstacle Points Project')
    ax.scatter(*obstacle_points_projection[-1], color='Silver', s=40)
    # New camera position
    ax.scatter(*cam_position, color='maroon', s=40, label='Cam Pose')
    
    ax.scatter(*tree_projection, color='gray', s=80, label='Tree center projection')
    ax.plot([intersect_a[0], intersect_b[0]], [intersect_a[1], intersect_b[1]], [intersect_a[2], intersect_b[2]], color='blue', linestyle='--',  label='Workspace Bound Line')
    

    ax.scatter(Obs_intersecting[:,0], Obs_intersecting[:,1], Obs_intersecting[:,2], color='green',s=40,  label='VP bound')
    ax.plot([center[0], Obs_intersecting[0][0]], [center[1], Obs_intersecting[0][1]], [center[2], Obs_intersecting[0][2]], 'k--', label='Line to obstacles')
    ax.plot([center[0], Obs_intersecting[1][0]], [center[1], Obs_intersecting[1][1]], [center[2], Obs_intersecting[1][2]], 'k--')
    # ax.plot([center[0], Obs_intersecting[0]], [center[1], Obs_intersecting[1]], [center[2], Obs_intersecting[2]], 'k--', label='Line to obstacles')
    ax.plot([obstacle_points_projection[0][0], obstacle_points_projection[1][0]], [obstacle_points_projection[0][1], obstacle_points_projection[1][1]], [obstacle_points_projection[0][2], obstacle_points_projection[1][2]], color='black',  linestyle='-', label='Obstacle')
    
    # Plotting the axis line
    axis_line = np.array([center  , center + axis])
    ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], color='blue', linewidth=2, label='Axis Line')
    
    # Plot lines from center to the first and last point
    ax.plot([center[0], intersect_a[0]], [center[1], intersect_a[1]], [center[2], intersect_a[2]], 'r--', label='Picking range line')
    ax.plot([center[0], intersect_b[0]], [center[1], intersect_b[1]], [center[2], intersect_b[2]],'r--' )
    # ax.plot([center[0], vp_points[-1, 0]], [center[1], vp_points[-1, 1]], [center[2], vp_points[-1, 2]], 'o--', label='Picking range line')

    # Create a filled area under the circle
    if auxiliary_plane:
        # x_fill = vp_points[:, 0]
        # y_fill = vp_points[:, 1]
        # z_fill = vp_points[:, 2]
        # x_fill = vp_points[workspace_in_range, 0]
        # y_fill = vp_points[workspace_in_range, 1]
        # z_fill = vp_points[workspace_in_range, 2]
        # vp_points[workspace_in_range, 0], vp_points[workspace_in_range , 1], vp_points[workspace_in_range , 2]
        # vp_points[~in_range & ~workspace_in_range, 2]
        # ax.plot_trisurf(x_fill, y_fill, z_fill, color='yellow', alpha=0.3)
        
        # ax.plot_trisurf(x_fill, y_fill, z_fill, color='Navy', alpha=0.3)
        # ax.plot_trisurf(vp_points[in_range, 0], vp_points[in_range, 1], vp_points[in_range, 2], color='grey', alpha=0.3)
        ax.plot_trisurf(vp_points[workspace_in_range, 0], vp_points[workspace_in_range, 1], vp_points[workspace_in_range, 2], color='Navy', alpha=0.3)
        ax.plot_trisurf(vp_points[~workspace_in_range, 0], vp_points[~workspace_in_range, 1], vp_points[~workspace_in_range, 2], color='yellow', alpha=0.3)
        
        pass
    
    # Setting equal aspect ratio in all axes for proper visualization of the circle
    max_range = np.array([vp_points[:,0].max()-vp_points[:,0].min(), vp_points[:,1].max()-vp_points[:,1].min(), vp_points[:,2].max()-vp_points[:,2].min()]).max() / 2.0
    mid_x = (vp_points[:,0].max()+vp_points[:,0].min()) * 0.5
    mid_y = (vp_points[:,1].max()+vp_points[:,1].min()) * 0.5
    mid_z = (vp_points[:,2].max()+vp_points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if plot:
        ax.legend()
        plt.show()

    # Dictionary to hold all points of interest
    points_dict = {
        # "intersect_pt": extended_point.tolist(),  # Void points are the intersection range points
        "intersect_pt": list(),                     # Void points are the intersection range points
        "first_pt": vp_points[0].tolist(),          # First point
        "end_pt": vp_points[-1].tolist(),           # End point
        "cam_pt": cam_position.tolist(),            # Camera position
        "obstacle_pt": obstacle_points.tolist()     # Given points (obstacles)
    }
    return vp_points




# Define the circle parameters
center, axis , radius= estimate_fruit_geo_info()
center = np.array([0, 0, 0])
radius = 3
axis = np.array([0, 0, 1])
# axis = np.array([1, 0, 1])
# axis = np.array([0, 1, 1])
nb_vp_samples= 100
# Useless Arg
max_angle = 180  # Max angle in degrees
# Obstacle points
obstacle_points = np.array([[1.5, 1.5, 0], [-1.5, 1.5, 0]])
# New point position
cam_position = np.array([0, 3, 0])
draw_circle(cam_position,center,obstacle_points,axis,radius,max_angle, nb_vp_samples)

