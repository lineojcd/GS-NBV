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
    print("orthogonal_vector and  second_vector are", first_vector, second_vector)
    return first_vector, second_vector

def point_Circle_plane_projection(point, center, axis):
    return point - np.dot(point - center, axis) * axis

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

def get_point_angle(query_pt, center, first_pt):
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
        # print("In")
        angle_radians += np.pi
        
    # Optionally convert the angle to degrees
    if False:
        angle_degrees = np.degrees(angle_radians)
    return angle_radians

def check_points_on_cicle(pointa, center, radius = 3):
    # Calculate the distance using numpy's linear algebra norm function
    distance = np.linalg.norm(pointa - center)
    on_circle = abs(distance - radius) <= 0.01
    return on_circle

# Draw a circle
def get_pose_info(tree_center, cam_position, center, obstacle_points,axis, radius, verbose = True, samples=360):
    # Normalize the axis vector
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    # Create auxiliary vectors to axis
    first_vector, second_vector = get_auxiliary_vector(axis)
    
    # Generate points on the circle
    theta = np.linspace(0, 2* np.pi, samples, endpoint=True)  # Ensures points are evenly spaced
    vp_pts = np.array([center + radius * (np.cos(t) * second_vector + np.sin(t) * first_vector) for t in theta])
    
    

    # get tree center auxiliary points
    tree_center_aux =  tree_center + np.array([1, 0, 0])
    # Calculate the projection of the tree_center onto the plane
    tree_projection = point_Circle_plane_projection(tree_center, center, axis)
    tree_aux_projection = point_Circle_plane_projection(tree_center_aux, center, axis)
    
    ws_inters_a, ws_inters_b = find_line_circle_intersect( tree_projection, tree_aux_projection, center,radius)
    # check intersect_a and intersect_b
    resa = check_points_on_cicle(ws_inters_a, center)
    resb = check_points_on_cicle(ws_inters_b, center)
    
    # Conduct obstacle_points project
    obs_pts_proj = []
    for pt in obstacle_points:
        v = point_Circle_plane_projection(pt, center, axis)
        obs_pts_proj.append(v)
    obs_pts_proj = np.array(obs_pts_proj)
    
    # Plot lines from obstacle points to center and their intersection points
    obs_inters = []
    for pt in obs_pts_proj:
        # Compute intersection points
        dir = pt - center
        # Normalize point vector
        v = center + dir / np.linalg.norm(dir) * radius
        obs_inters.append(v)
    obs_inters = np.array(obs_inters)
    
    # Calculate intersection point angles
    first_pt = vp_pts[0]
    obs_min_angle = get_point_angle(obs_inters[0], center, first_pt)
    obs_max_angle = get_point_angle(obs_inters[-1], center, first_pt)
    # Determine which points are within the intersection angle range
    obs_in_range = (theta >= obs_min_angle) & (theta <= obs_max_angle)
    
    # Calculate intersection angles and find range
    # ws_inters = np.array([ws_inters_a, ws_inters_b])
    ws_min_angle =  get_point_angle(ws_inters_a, center, first_pt)
    ws_max_angle =  get_point_angle(ws_inters_b, center, first_pt)
    # Determine which points are within the workspace intersection angle range
    ws_in_range = (theta >= ws_min_angle) & (theta <= ws_max_angle)
    # print("theta is ", theta)
    # print("ws_in_range is ", ws_in_range)
    
    if verbose:
        print("orthogonal_vector and second_vector are", first_vector, second_vector)
        print("Starting Pt is ",vp_pts[0])
        print("len of theta and vp_points:", len(theta) ,len(vp_pts) )
        print("tree_projection and tree_aux_projection are ",tree_projection, tree_aux_projection)
        print("WorkSpace intersection pta on circle:", resa)
        print("WorkSpace intersection ptb on circle:", resb)
        print("Obstacle_points are: ",obstacle_points)
        print("Obstacle_points_projection are: ",obs_pts_proj)
        print("Obstacle intersecting points coords: ", obs_inters)
        print("Obstacle intersecting points min and max angle are: ", obs_min_angle, obs_max_angle)
        print("workspace_min_angle and  workspace_max_angle are: ", ws_min_angle, ws_max_angle)
    
    # Dictionary to hold all points of interest
    points_dict = {
        "fruit_center": center,
        "fruit_axis": axis,
        "cam_position": cam_position,
        "vp_pts": vp_pts,
        "tree_projection": tree_projection,
        "workspace_bound_pta": ws_inters_a,
        "workspace_bound_ptb": ws_inters_b,
        "obstacle_pts": obstacle_points,
        "obstacle_pts_projection": obs_pts_proj,
        "obstacle_pts_intersect_on_circle": obs_inters,
        "obstacle_in_range":obs_in_range,
        "workspace_in_range":ws_in_range,
        "last_vp_bound": None,
        "last_fruit_center": None
    }
    return points_dict

def visualize(dictionary, auxiliary_plane=True):
    center = dictionary["fruit_center"]
    axis = dictionary["fruit_axis"]
    cam_position = dictionary["cam_position"]
    vp_pts = dictionary["vp_pts"]
    first_pt = vp_pts[0]
    tree_projection = dictionary["tree_projection"]
    ws_inters_a = dictionary["workspace_bound_pta"]
    ws_inters_b = dictionary["workspace_bound_ptb"]
    obs_pts = dictionary["obstacle_pts"]
    obs_pts_proj = dictionary["obstacle_pts_projection"]
    obs_inters = dictionary["obstacle_pts_intersect_on_circle"]
    obs_in_range = dictionary["obstacle_in_range"]
    ws_in_range = dictionary["workspace_in_range"]
    axis_line_pt = np.array([center  , center + axis])
    last_vp_bound = dictionary["last_vp_bound"]
    last_center = dictionary["last_fruit_center"]
    
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Color key points accordingly
    ax.scatter(*center, color='red', s=80, label='Fruit Center')
    ax.scatter(*first_pt, s=100,color='magenta', label='First Pose')
    ax.scatter(vp_pts[obs_in_range, 0], vp_pts[obs_in_range , 1], vp_pts[obs_in_range , 2], color='grey', label='Void Pose')
    ax.scatter(vp_pts[ws_in_range, 0], vp_pts[ws_in_range , 1], vp_pts[ws_in_range , 2], color='Navy', label='Unreachable Pose')
    ax.scatter(vp_pts[~obs_in_range & ~ws_in_range, 0], vp_pts[~obs_in_range & ~ws_in_range, 1], vp_pts[~obs_in_range & ~ws_in_range, 2], color='magenta', label='Valid Pose')
    ax.scatter(*ws_inters_a, color='cyan',s=40,  label='Workspace Bound Points')
    ax.scatter(*ws_inters_b, color='cyan',s=40)
    ax.scatter(*obs_pts[0], color='Black', s=40, label='Obstacle Points')
    ax.scatter(*obs_pts[-1], color='Black', s=40)
    ax.scatter(*obs_pts_proj[0], color='Silver', s=40,label='Obstacle Points Project')
    ax.scatter(*obs_pts_proj[-1], color='Silver', s=40)
    ax.scatter(*cam_position, color='Purple', s=80, label='Cam Pose')
    ax.scatter(*tree_projection, color='gray', s=80, label='Tree center projection')
    ax.scatter(obs_inters[:,0], obs_inters[:,1], obs_inters[:,2], color='Lime',s=40,  label='VP bound')
    ax.plot([ws_inters_a[0], ws_inters_b[0]], [ws_inters_a[1], ws_inters_b[1]], [ws_inters_a[2], ws_inters_b[2]], color='blue', linestyle='--',  label='Workspace Bound Line')
    ax.plot([center[0], obs_inters[0][0]], [center[1], obs_inters[0][1]], [center[2], obs_inters[0][2]], 'k--', label='Line to obstacles')
    ax.plot([center[0], obs_inters[1][0]], [center[1], obs_inters[1][1]], [center[2], obs_inters[1][2]], 'k--')
    ax.plot([obs_pts_proj[0][0], obs_pts_proj[1][0]], [obs_pts_proj[0][1], obs_pts_proj[1][1]], [obs_pts_proj[0][2], obs_pts_proj[1][2]], color='black',  linestyle='-', label='Obstacle Range')
    ax.plot(axis_line_pt[:, 0], axis_line_pt[:, 1], axis_line_pt[:, 2], color='blue', linewidth=2, label='Fruit Central Axis')
    # Plot lines from center to the first and last valid point
    ax.plot([center[0], ws_inters_a[0]], [center[1], ws_inters_a[1]], [center[2], ws_inters_a[2]], 'r--', label='Picking range line')
    ax.plot([center[0], ws_inters_b[0]], [center[1], ws_inters_b[1]], [center[2], ws_inters_b[2]],'r--' )

    # Create a filled area under the circle
    if auxiliary_plane:
        ax.plot_trisurf(vp_pts[ws_in_range, 0], vp_pts[ws_in_range, 1], vp_pts[ws_in_range, 2], color='Navy', alpha=0.3)
        ax.plot_trisurf(vp_pts[~ws_in_range, 0], vp_pts[~ws_in_range, 1], vp_pts[~ws_in_range, 2], color='yellow', alpha=0.3)
    
    if last_vp_bound is not None:
        ax.scatter(last_vp_bound[:,0], last_vp_bound[:,1], last_vp_bound[:,2], color='green',s=40,  label='VP bound iter1')    
        ax.plot([center[0], last_vp_bound[0][0]], [center[1], last_vp_bound[0][1]], [center[2], last_vp_bound[0][2]], 'y--', label='Line to obstacles iter1')
        ax.plot([center[0], last_vp_bound[1][0]], [center[1], last_vp_bound[1][1]], [center[2], last_vp_bound[1][2]], 'y--')
        ax.scatter(*last_center, color='Maroon', s=80, label='Fruit Center iter1')
        # Plot here for convenience
        ax.scatter(*np.array([0, 3, 0]), color='Olive', s=80, label='Cam Pose iter1')
        # Generate points for the circle
        theta = np.linspace(0, 2 * np.pi, 100)
        x = last_center[0] + radius * np.cos(theta)  # X coordinates
        y = last_center[1] + radius * np.sin(theta)  # Y coordinates
        z = last_center[2] + np.zeros_like(theta)    # Z coordinates are all zero, as the circle is in the XY plane
        ax.plot(x, y, z, label='Circle in XY plane')  # Plot the circle

    # Setting equal aspect ratio in all axes for proper visualization of the circle
    max_range = np.array([vp_pts[:,0].max()-vp_pts[:,0].min(), vp_pts[:,1].max()-vp_pts[:,1].min(), vp_pts[:,2].max()-vp_pts[:,2].min()]).max() / 2.0
    mid_x = (vp_pts[:,0].max()+vp_pts[:,0].min()) * 0.5
    mid_y = (vp_pts[:,1].max()+vp_pts[:,1].min()) * 0.5
    mid_z = (vp_pts[:,2].max()+vp_pts[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    # plt.show()
    return fig
    
    
if __name__ == '__main__':
    # Define the circle parameters
    center = np.array([0, 0, 0])
    radius = 1
    axis = np.array([0, 0, 1])
    nb_vp_samples= 100

    # Useless Arg
    max_angle = 180  # Max angle in degrees
    # Obstacle points
    obstacle_points = np.array([[1.5, 1.5, 0], [-1.5, 1.5, 0]])
    # New point position
    cam_position = np.array([0, 3, 0])
    tree_center =  np.array([-1, -1, 1])
    # draw_circle(tree_center, cam_position,center,obstacle_points,axis,radius,max_angle, nb_vp_samples)


    ################
    center_1 = np.array([0, 0, 0])
    axis_1 = np.array([0, 0, 1])
    cam_posi_1 = np.array([0, 2, 0])
    # center_2 = np.array([0, -0.5, 0])
    # axis_2 = np.array([0, 0, 1])
    # cam_posi_2 = np.array([3, 0, 0])
    radius = 0.27
    tree_center =  np.array([-1, -1, 1])

    obstacle_points = np.array([[1.5, 1.5, 0], [-1.5, 1.5, 0]])
    # obstacle_points = np.array([[ 0.48908,-0.14105,1.1822], [0.48346,-0.14092,1.1742]])
    # center_1 =  np.array([0.4939, -0.2650,  1.1801])
    # tree_center =  np.array([0.5, -0.4, 0.66])
    for i in range(1):
        print("iter :", i)
        pts_dict = get_pose_info(tree_center, cam_posi_1, center_1, obstacle_points,axis_1, radius)
        # pts_dict2 = get_pose_info(tree_center, cam_posi_2, center_2, obstacle_points,axis_2, radius)
        # if pts_dict2["last_vp_bound"] is None:
        #     pts_dict2["last_vp_bound"]  = pts_dict["obstacle_pts_intersect_on_circle"]     
        #     pts_dict2["last_fruit_center"]  = pts_dict["fruit_center"]     
        # fig = visualize(pts_dict2)
        
        fig = visualize(pts_dict)
        plt.show()


    if False:
        centers = np.array([center_1, center_2])
        axis = np.array([axis_1, axis_2])
        cam_posi = np.array([cam_posi_1, cam_posi_2])
        for i in range(2):
            print("iter :", i)
            pts_dict = get_pose_info(tree_center, cam_posi[i], centers[i], obstacle_points,axis[i], radius)
            visualize(pts_dict)
    
    
    
    