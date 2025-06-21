import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MyMethod:
    def __init__(self):
        # self.f_pos = np.array([0, 0, 0])
        # self.f_axis = np.array([0, 0, 1])
        # self.radius = 0.27
        self.show_plot = False
        self.verbose_mysampler = False
        self.workspace_bound_a = 225            # degree
        self.workspace_bound_b = 315            # degree
        self.num_samples = 4
    
    def get_vp_from_ring(self, f_pos, f_axis, radius, samples=360):
        # Normalize the axis vector
        axis = np.array(f_axis)
        axis = axis / np.linalg.norm(f_axis)
        # Create auxiliary vectors to axis
        first_vector, second_vector = self.get_auxiliary_vector(f_axis)
        
        # Generate points on the circle
        # theta = np.linspace(0, 2* np.pi, samples - 1, endpoint=True)  # Ensures points are evenly spaced
        theta = np.linspace(0, 2* np.pi, samples -1 , endpoint=False)  # Ensures points are evenly spaced
        # theta = np.linspace(0, 2* np.pi, 4 , endpoint=False)  # Ensures points are evenly spaced
        vp_from_ring = np.array([f_pos + radius * (np.cos(t) * second_vector + np.sin(t) * first_vector) for t in theta])
        if self.verbose_mysampler:
            print("vp_from_ring is:", vp_from_ring)
            print("len of vp_from_ring", len(vp_from_ring))
            print("theta is:", theta)
            print(" theta", len(theta))
        return vp_from_ring, theta

    def get_auxiliary_vector(self, axis):
        # Create an orthogonal vector to axis
        if (axis == np.array([1, 0, 0])).all():
            first_vector = np.cross(axis, np.array([0, 1, 0]))
        else:
            first_vector = np.cross(axis, np.array([1, 0, 0]))
        first_vector = first_vector / np.linalg.norm(first_vector)
        second_vector = np.cross(axis, first_vector)
        second_vector = -1* second_vector / np.linalg.norm(second_vector) +np.array([0, 0, 0])
        if self.verbose_mysampler:
            print("orthogonal_vector and  second_vector are", first_vector, second_vector)
        return first_vector, second_vector

    def degree_to_radian(self, degree):
        return np.radians(degree)

    def filter_nonworkspace_point(self, pts, thetas):
        # Convert workspace bounds from degrees to radians
        ws_bound_a_rad = self.degree_to_radian(self.workspace_bound_a)
        ws_bound_b_rad = self.degree_to_radian(self.workspace_bound_b)
        
        # Filter the points and angles that are outside the range [workspace_bound_a_rad, workspace_bound_b_rad]
        filtered_vp = []
        filtered_theta = []
        
        for i, rad in enumerate(theta):
            if not (ws_bound_a_rad <= rad <= ws_bound_b_rad):
                filtered_vp.append(pts[i])
                filtered_theta.append(rad)

        filtered_vp = np.array(filtered_vp)
        filtered_theta = np.array(filtered_theta)

        if self.verbose_mysampler:
            print("Filtered vp_from_ring:", filtered_vp)
            print("Filtered theta (radians):", filtered_theta)
        return filtered_vp, filtered_theta
    
    def point_Circle_plane_projection(self, point, center, axis):
        return point - np.dot(point - center, axis) * axis

    def get_obstacle_pts_projection(self,obstacle_list, f_pos, f_axis):
        obs_pts_proj = []
        for pt in obstacle_list:
            v = self.point_Circle_plane_projection(pt, f_pos, f_axis)
            obs_pts_proj.append(v)
        obs_pts_proj = np.array(obs_pts_proj)
    
    def compute_line_circle_intersection(self,pt, center, radius):
        # Compute intersection points
        dir = pt - center
        # Normalize point vector
        intersect_pt = center + dir / np.linalg.norm(dir) * radius
        return intersect_pt
    
    def get_obstacle_pts_intersection_on_circle(self, obstacle_list, center, radius):
        obs_intersect_pts = []
        for pt in obstacle_list:
            obs_intersect_pts.append(self.compute_line_circle_intersection(pt, center, radius))
        obs_intersect_pts = np.array(obs_intersect_pts)
        return obs_intersect_pts
    
    def get_point_angle(self, query_pt, center, first_pt):
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
        return angle_radians
        
    def get_obstacle_pts_intersection_angle_on_circle(self, obstacle_list, center, first_pt):
        obs_intersect_pts_angle = []
        # print("obstacle_list", obstacle_list)
        for pt in obstacle_list:
            pt_rad = self.get_point_angle(pt, center, first_pt)
            # print("pt_rad:", pt_rad)
            obs_intersect_pts_angle.append(pt_rad)
        obs_intersect_pts_angle = np.array(obs_intersect_pts_angle)
        print("obs_intersect_pts_angle", obs_intersect_pts_angle)
        return obs_intersect_pts_angle
    
    def filter_obstacle_point(self, obs_intersect_pts_angle, center, radius, filtered_vp, filtered_theta,first_pt, first_theta):
        # Filter the points and angles that are inside of the obstacle range
        filtered_obs_vp = []
        filtered_obs_theta = []
        for k, rad in enumerate(filtered_theta):
            # Iterate over the obstacle_list and bring 2 elements in each iteration
            for i in range(0, len(obs_intersect_pts_angle), 2):
                obs_min_rad = obs_intersect_pts_angle[i]
                obs_max_rad = obs_intersect_pts_angle[i+1]
                if not (obs_min_rad <= rad <= obs_max_rad):
                    filtered_obs_vp.append(filtered_vp[k])
                    filtered_obs_theta.append(filtered_theta[k])
                    break
        filtered_obs_vp = np.array(filtered_obs_vp)
        filtered_obs_theta = np.array(filtered_obs_theta)
        return filtered_obs_vp, filtered_obs_theta
    
    def vp_reorder(self, vplist, thetalist):
        # Define the boundaries
        lower_bound = 1.5 * np.pi  # 1.5π in radians
        upper_bound = 2 * np.pi    # 2π in radians
        
        # Find the indices for the two parts
        first_part_idxs = np.where((thetalist >= lower_bound) & (thetalist <= upper_bound))[0]
        second_part_idxs = np.where(thetalist < lower_bound)[0]
        
        # Reorder the indices
        reordered_idxs = np.concatenate((first_part_idxs, second_part_idxs))
        # Reorder both filtered_obs_theta and filtered_obs_vp
        reordered_theta = thetalist[reordered_idxs]
        reordered_vp = vplist[reordered_idxs]
        if self.verbose_mysampler:
            print("reordered_vp:", reordered_vp)
        return reordered_vp,reordered_theta

    def uniform_sampling_on_validvps(self, valid_vplist):
        # Calculate the indices to equally sample 4 points
        idxs = [int(i * (len(valid_vplist) - 1) / (self.num_samples - 1)) for i in range(self.num_samples)]
        # Get the corresponding points from the list
        selected_vplist = np.array([valid_vplist[i] for i in idxs])
        # filtered_obs_theta = np.array(filtered_obs_theta)
        return selected_vplist
    
    
    
    def visulize(self, f_pos, f_axis, radius, camera, vp_pts, selected_vps, obs_pts_on_cicle, obs_pts_proj):
        # Visualization of vp_pts
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the generated points
        ax.scatter(vp_pts[:, 0], vp_pts[:, 1], vp_pts[:, 2], color='yellow', label='vp_pts', alpha=0.1)
        ax.scatter(selected_vps[:, 0], selected_vps[:, 1], selected_vps[:, 2], color='magenta', label='selected_vps', alpha=1)
        ax.scatter(obs_pts_on_cicle[:, 0], obs_pts_on_cicle[:, 1], obs_pts_on_cicle[:, 2], color='red', label='obstacle_pts_on_circle')
        ax.scatter(obs_pts_proj[:, 0], obs_pts_proj[:, 1], obs_pts_proj[:, 2], color='grey', label='obstacle_pts_projection')

        # Plot the axis vector from f_pos in the direction of f_axis
        ax.quiver(f_pos[0], f_pos[1], f_pos[2], f_axis[0], f_axis[1], f_axis[2],
                    color='r', label='f_axis', length=radius, normalize=True)
        
        # Plot f_pos as a red dot
        ax.scatter(f_pos[0], f_pos[1], f_pos[2], color='r', s=100, label='f_pos')
        ax.scatter(camera[0], camera[1], camera[2], color='b', s=100, label='camera_proj')

        # Set labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # Set the aspect ratio to be equal for all axes
        ax.set_box_aspect([1,1,1])
        # Set the limits of the plot
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])
        # Show the plot
        plt.legend()
        plt.show()
    
if __name__ == '__main__':
    f_pos = np.array([0, 0, 0])
    f_axis = np.array([0, 0, 1])
    radius = 0.27
    camera = np.array([0, 0.3, 0.2])
    obstacle_list = [np.array([0.05, 0.05, 0.2]), np.array([-0.05, 0.05, 0.2])]
    
    met = MyMethod()
    vp_from_ring, theta = met.get_vp_from_ring(f_pos, f_axis, radius)
    first_pt = vp_from_ring[0]
    first_theta = theta[0]
    
    filtered_vp, filtered_theta = met.filter_nonworkspace_point(vp_from_ring, theta)
    # Conduct obstacle_points project
    obs_pts_proj = met.get_obstacle_pts_projection(obstacle_list, f_pos, f_axis)
    # Calculate intersection point angles
    obs_intersect_pts = met.get_obstacle_pts_intersection_on_circle(obs_pts_proj, f_pos, radius)
    # Calculate intersection point angles
    obs_intersect_pts_angle = met.get_obstacle_pts_intersection_angle_on_circle(obs_intersect_pts, f_pos, first_pt)
        
    filtered_obs_vp, filtered_obs_theta = met.filter_obstacle_point(obs_intersect_pts_angle, f_pos, radius, filtered_vp, filtered_theta, first_pt, first_theta)
    
    
    
    
    camera_on_circle = met.point_Circle_plane_projection(camera, f_pos, f_axis)
    print("point_on_circle is:",camera_on_circle)
    
    # print("filtered_obs_theta is:",filtered_obs_theta)
    print("len of filtered_obs_vp", len(filtered_obs_vp))
    
    reordered_vp, reordered_theta = met.vp_reorder(filtered_obs_vp, filtered_obs_theta)
    uniform_sampling_on_validvps = met.uniform_sampling_on_validvps(reordered_vp)
    print("uniform_sampling_on_validvps", uniform_sampling_on_validvps)
    
    # visulize
    met.visulize(f_pos, f_axis, radius, camera_on_circle, filtered_obs_vp,uniform_sampling_on_validvps, obs_intersect_pts, obs_pts_proj)
    