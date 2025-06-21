# ROS node to visualize topics in rviz

import rospy
import numpy as np
import ros_numpy
import struct
import math
from copy import deepcopy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point, Pose, PoseArray
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import time

class MyRvizVisualizer:
    def __init__(self):
        # Frames
        self.world_frame_id  = "world"
        self.camera_frame_id  = "camera_color_optical_frame"
        self.octomap_frame_ns = "voxel_frame"
        self.viewpoint_ns = "viewpoints"
        self.candidate_viewpoint_ns = "candidate_view_samples"
        # self.octomap_frame_x_axis = "voxel_frame"
        self.avo_height_offset = rospy.get_param("avo_height_offset", "0.029")
        self.picking_ring_color = [int(x) for x in rospy.get_param('picking_ring_color').split()]
        self.fruit_voxel_color = [int(x) for x in rospy.get_param('fruit_voxel_color').split()]
        self.peduncle_voxel_color = [int(x) for x in rospy.get_param('peduncle_voxel_color').split()]
        self.background_voxel_color = [int(x) for x in rospy.get_param('background_voxel_color').split()]
        self.fruit_posi_GT = [float(x) for x in rospy.get_param('fruit_posi_GT').split()]
        self.fruit_axis_GT = [float(x) for x in rospy.get_param('fruit_axis_GT').split()]
        self.fruit_GT_color = [float(x) for x in rospy.get_param('fruit_GT_color').split()]
        self.fruit_axis_scale = [float(x) for x in rospy.get_param('fruit_axis_scale').split()]
        self.fruit_position_scale = rospy.get_param("fruit_position_scale", "0.01")
        self.fruit_EST_color = [float(x) for x in rospy.get_param('fruit_EST_color').split()]
        self.valid_viewpoint_color = [float(x) for x in rospy.get_param('valid_viewpoint_color').split()]
        self.valid_viewpoint_scale = [float(x) for x in rospy.get_param('valid_viewpoint_scale').split()]
        self.valid_viewpoint_length_coef = rospy.get_param('valid_viewpoint_length_coef', "0.01")
        self.new_viewpoint_color = [float(x) for x in rospy.get_param('new_viewpoint_color').split()]
        self.new_viewpoint_scale = [float(x) for x in rospy.get_param('new_viewpoint_scale').split()]
        self.new_viewpoint_length_coef = rospy.get_param('new_viewpoint_length_coef', "0.05")
        self.reachable_viewpoint_color = [float(x) for x in rospy.get_param('reachable_viewpoint_color').split()]
        self.filtered_viewpoint_color = [float(x) for x in rospy.get_param('filtered_viewpoint_color').split()]
        self.previous_viewpoint_color = [float(x) for x in rospy.get_param('previous_viewpoint_color').split()]
        self.previous_viewpoint_scale = [float(x) for x in rospy.get_param('previous_viewpoint_scale').split()]
        self.previous_viewpoint_length_coef = rospy.get_param('previous_viewpoint_length_coef', "0.05")
        self.nbv_viewpoint_color = [float(x) for x in rospy.get_param('nbv_viewpoint_color').split()]
        self.nbv_viewpoint_scale = [float(x) for x in rospy.get_param('nbv_viewpoint_scale').split()]
        self.nbv_viewpoint_length_coef = rospy.get_param('nbv_viewpoint_length_coef', "0.01")
        self.octomap_frame_scale = [float(x) for x in rospy.get_param('octomap_frame_scale').split()]
        self.vp_sampling_radius = rospy.get_param("vp_sampling_radius", "0.24")
        self.verbose_rviz = rospy.get_param("verbose_rviz")
        self.voxel_origin = [float(x) for x in rospy.get_param('voxel_origin').split()]
        self.voxel_frame_length = rospy.get_param("voxel_frame_length", "0.1")
        # Initialize the rviz visualizer
        self.octomap_frame_pub = rospy.Publisher('octomap_frame', MarkerArray, queue_size=1)
        self.reachable_viewpoints_pub = rospy.Publisher("reachable_viewpoints", MarkerArray, queue_size=1)
        self.filtered_viewpoints_pub = rospy.Publisher("filtered_viewpoints", MarkerArray, queue_size=1)
        self.new_viewpoints_pub = rospy.Publisher("new_viewpoints", MarkerArray, queue_size=1)
        self.previous_viewpoints_pub = rospy.Publisher("previous_viewpoints", MarkerArray, queue_size=1)
        self.valid_viewpoints_pub = rospy.Publisher("valid_viewpoints", MarkerArray, queue_size=1)
        self.nbv_viewpoint_pub = rospy.Publisher("nbv_viewpoint", MarkerArray, queue_size=1)
        self.voxels_pc2_pub = rospy.Publisher("voxels", PointCloud2, queue_size=1)
        self.ROI_voxels_pc2_pub = rospy.Publisher("roi_voxels", PointCloud2, queue_size=1)
        self.rois_pub = rospy.Publisher("rois", MarkerArray, queue_size=1)
        self.camera_bounds_pub = rospy.Publisher("camera_bounds", Marker, queue_size=1)
        self.world_model_pub = rospy.Publisher("world_model/objects", MarkerArray, queue_size=1)
        self.point_cloud_pub = rospy.Publisher("gt_point_cloud", PointCloud2, queue_size=1)
        self.poses_with_covariance_pub = rospy.Publisher("poses_with_covariance", MarkerArray, queue_size=1)
        self.poses_pub = rospy.Publisher("pose_estimation/poses", PoseArray, queue_size=1)
        self.point_pub = rospy.Publisher("point", Marker, queue_size=1)
        self.class_ids_pub = rospy.Publisher("class_ids", MarkerArray, queue_size=1)
        self.picking_ring_GT_pub = rospy.Publisher("picking_ring", Marker, queue_size=1)
        self.curve_pub = rospy.Publisher("valid_pose", Marker, queue_size=1)
        self.points_pub = rospy.Publisher("points", Marker, queue_size=1)
        self.pred_points_pub = rospy.Publisher("pred_points", Marker, queue_size=1)
        self.point_cloud_pub2 = rospy.Publisher("true_point_cloud", PointCloud2, queue_size=1)
        self.gain_image_pub = rospy.Publisher("gain_image", Image, queue_size=1)
        self.fruit_pose_GT_pub = rospy.Publisher("fruit_pose_gt", MarkerArray, queue_size=1)
        self.fruit_pose_EST_pub = rospy.Publisher("fruit_pose_est", MarkerArray, queue_size=1)
        
    def create_axis_marker(self,id, color, scale, ns, start, end):
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [Point(*start), Point(*end)]    # Set start and end points
        marker.scale =  Vector3(scale[0], scale[1], scale[2])   # x Shaft diameter, y Head diameter, z Head length (not used)
        marker.color =  ColorRGBA(color[0], color[1], color[2], color[3])
        return marker
    
    def visualize_octomap_frame(self):
        # Publish each axis as a separate marker: X-axis in red, Y in green, Z in blue
        color_x = (1.0, 0.0, 0.0, 1.0)
        color_y = (0.0, 1.0, 0.0, 1.0)
        color_z = (0.0, 0.0, 1.0, 1.0)
        marker_array = MarkerArray()
        x_axis_marker = self.create_axis_marker(id=0, color=color_x, ns=self.octomap_frame_ns, scale=self.octomap_frame_scale,  start=self.voxel_origin, 
            end=(self.voxel_origin[0] + self.voxel_frame_length, self.voxel_origin[1], self.voxel_origin[2]) )
        y_axis_marker = self.create_axis_marker(id=1, color=color_y, ns=self.octomap_frame_ns, scale=self.octomap_frame_scale,  start=self.voxel_origin,
            end=(self.voxel_origin[0], self.voxel_origin[1] + self.voxel_frame_length, self.voxel_origin[2]) )
        z_axis_marker = self.create_axis_marker(id=2, color=color_z, ns=self.octomap_frame_ns, scale=self.octomap_frame_scale,  start=self.voxel_origin,
            end=(self.voxel_origin[0], self.voxel_origin[1], self.voxel_origin[2] + self.voxel_frame_length) )
        marker_array.markers.append(x_axis_marker)
        marker_array.markers.append(y_axis_marker)
        marker_array.markers.append(z_axis_marker)
        self.octomap_frame_pub.publish(marker_array)
        
    def visualize_viewpointslist(self, viewpoints, fruit_position, viewpoint_type):
        if len(viewpoints)==0:
            print("no viewpoint in the list")
            return 0
        marker_array = MarkerArray()
        if viewpoint_type == "previous_vps":
            color = self.previous_viewpoint_color
            scale = self.previous_viewpoint_scale
            length_coef = self.previous_viewpoint_length_coef
            publisher = self.previous_viewpoints_pub
        if viewpoint_type == "new_vps":
            color = self.new_viewpoint_color
            scale = self.new_viewpoint_scale
            length_coef = self.new_viewpoint_length_coef
            publisher = self.new_viewpoints_pub
        if viewpoint_type == "reachable_vps":
            color = self.reachable_viewpoint_color
            scale = self.new_viewpoint_scale
            length_coef = self.new_viewpoint_length_coef
            publisher = self.reachable_viewpoints_pub
        if viewpoint_type == "filtered_vps":
            color = self.filtered_viewpoint_color
            scale = self.new_viewpoint_scale
            length_coef = self.new_viewpoint_length_coef
            publisher = self.filtered_viewpoints_pub
        if viewpoint_type == "valid_vps":
            color = self.valid_viewpoint_color
            scale = self.valid_viewpoint_scale
            length_coef = self.valid_viewpoint_length_coef
            publisher = self.valid_viewpoints_pub
        if viewpoint_type == "nbv_vp":
            color = self.nbv_viewpoint_color
            scale = self.nbv_viewpoint_scale
            length_coef = self.nbv_viewpoint_length_coef
            publisher = self.nbv_viewpoint_pub
        for idx, vp in enumerate(viewpoints):
            dir_vec =  fruit_position - vp[:3]
            dir_vec /= np.linalg.norm(dir_vec)  # Normalize the vector
            dir_vec *= length_coef
            vp_marker = self.create_axis_marker(id=idx, color=color, scale=scale, ns=self.viewpoint_ns , start=vp[:3], end=vp[:3]+dir_vec)
            marker_array.markers.append(vp_marker)
        publisher.publish(marker_array)
    
    def get_pose_from_axis(self,axis):
        # Reference axis (Z-axis in this case)
        reference_axis = np.array([0, 0, 1])
        # Step 2: Compute the rotation axis (cross product)
        rotation_axis = np.cross(reference_axis, axis)
        # Step 3: Compute the angle (dot product and arccos)
        cos_theta = np.dot(reference_axis, axis)
        angle = np.arccos(cos_theta)
        # Step 4: Convert to quaternion
        if np.linalg.norm(rotation_axis) != 0:  # Check if the rotation axis is not zero
            rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
            quaternion = R.from_rotvec(rotation_axis_normalized * angle).as_quat()
        else:
            # If the direction is exactly aligned with the reference axis, no rotation is needed
            quaternion = np.array([0, 0, 0, 1])
        return quaternion

    def visualize_fruit_axis_EST(self,point, color, quaternion):
        ns = "fruit_axis_EST"
        id = 0
        scale =  Vector3(self.fruit_axis_scale[0], self.fruit_axis_scale[1], self.fruit_axis_scale[2]) 
        return self.visualize_axis(id, ns, scale, color, point, quaternion)
    
    def visualize_fruit_position_EST(self,point, color):
        id = 0
        ns = "fruit_position_EST"
        scale = Vector3(self.fruit_position_scale, self.fruit_position_scale, self.fruit_position_scale)
        return self.visualize_point(id, ns, scale, color, point)
    
    def visualize_fruit_pose_EST(self, fruit_posi_est, fruit_axis_est, test=False):
        quaternion = self.get_pose_from_axis(fruit_axis_est)
        marker_array = MarkerArray()
        point = np.array(fruit_posi_est)
        color =  ColorRGBA(self.fruit_EST_color[0], self.fruit_EST_color[1], self.fruit_EST_color[2], self.fruit_EST_color[3])
        axis_marker = self.visualize_fruit_axis_EST(point, color, quaternion)
        sphere_marker = self.visualize_fruit_position_EST(point, color)
        marker_array.markers.append(axis_marker)
        marker_array.markers.append(sphere_marker)
        self.fruit_pose_EST_pub.publish(marker_array)

    def visualize_axis(self, id, ns, scale, color, point, pose):
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.CYLINDER  
        marker.action = Marker.ADD
        marker.pose.position = Point(*point)
        marker.pose.orientation.x = pose[0]
        marker.pose.orientation.y = pose[1]
        marker.pose.orientation.z = pose[2]
        marker.pose.orientation.w = pose[3]
        marker.scale =  scale
        marker.color =  color
        marker.lifetime = rospy.Duration()                  # Set the lifetime of the marker
        return marker       

    def visualize_point(self, id, ns, scale, color, point):
        sphere_marker = Marker()
        sphere_marker.header.frame_id = self.world_frame_id
        sphere_marker.header.stamp = rospy.Time.now()
        sphere_marker.ns = ns
        sphere_marker.id = id
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position = Point(*point)
        sphere_marker.pose.orientation.w = 1.0
        sphere_marker.scale = scale  
        sphere_marker.color = color
        sphere_marker.lifetime = rospy.Duration()
        return sphere_marker
        
    def visualize_fruit_axis_GT(self,point, color):
        ns = "fruit_axis_GT"
        id = 0
        rospy.logwarn("visualize_fruit_pose_GT assume the axis is stright upward")
        pose = (0.0, 0.0, 0.0, 1.0)
        scale =  Vector3(self.fruit_axis_scale[0], self.fruit_axis_scale[1], self.fruit_axis_scale[2]) 
        return self.visualize_axis(id, ns, scale, color, point, pose)
    
    def visualize_fruit_position_GT(self,point, color):
        id = 0
        ns = "fruit_position_GT"
        scale = Vector3(self.fruit_position_scale, self.fruit_position_scale, self.fruit_position_scale)
        return self.visualize_point(id, ns, scale, color, point)
    
    def visualize_fruit_pose_GT(self):
        marker_array = MarkerArray()
        point = np.array(self.fruit_posi_GT)
        color =  ColorRGBA(self.fruit_GT_color[0], self.fruit_GT_color[1], self.fruit_GT_color[2], self.fruit_GT_color[3])
        axis_marker = self.visualize_fruit_axis_GT(point, color)
        sphere_marker = self.visualize_fruit_position_GT(point, color)
        marker_array.markers.append(axis_marker)
        marker_array.markers.append(sphere_marker)
        self.fruit_pose_GT_pub.publish(marker_array)
        
    def visualize_voxels(self, points: np.array, class_ids: np.array) -> None:
        """
        Visualize the voxels in Rviz as a PointCloud2 message
        :param points: (N, 3) array of points
        :param class_ids: (N, 1) array of semantics
        """
        assert points.shape[1] == 3, "The input array must have shape (N, 3)"
        # color_contour = struct.unpack("I", struct.pack("BBBB", 255, 0, 255, 255))[0]    #Magenta BGR
        # color_envelope = struct.unpack("I", struct.pack("BBBB", 255,255,0, 255))[0]     #Cyan    BGR  
        color_background = struct.unpack("I", struct.pack("BBBB", self.background_voxel_color[0],self.background_voxel_color[1],self.background_voxel_color[2], self.background_voxel_color[3]))[0]              
        color_fruit = struct.unpack("I", struct.pack("BBBB", self.fruit_voxel_color[0],self.fruit_voxel_color[1],self.fruit_voxel_color[2], self.fruit_voxel_color[3]))[0]              
        color_peduncle = struct.unpack("I", struct.pack("BBBB", self.peduncle_voxel_color[0],self.peduncle_voxel_color[1],self.peduncle_voxel_color[2], self.peduncle_voxel_color[3]))[0]              
        
        # Create a structured NumPy array with named fields
        points_dtype = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32),("rgb", np.uint32),])
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = color_background
        
        for i in range(points.shape[0]):
            if class_ids[i] == 0:
                points_arr["rgb"][i] = color_fruit
            if class_ids[i] == 1:
                points_arr["rgb"][i] = color_peduncle
            if class_ids[i] == 2:
                points_arr["rgb"][i] = color_fruit
            if class_ids[i] == 3:
                points_arr["rgb"][i] = color_background

        # Convert the NumPy array to a PointCloud2 message
        voxel_points = ros_numpy.point_cloud2.array_to_pointcloud2(points_arr, rospy.Time.now(), self.world_frame_id)
        self.voxels_pc2_pub.publish(voxel_points)

    def visualize_point_cloud(self, points: np.array, color: np.array) -> None:
        """
        Visualize the point cloud in Rviz as a PointCloud2 message
        :param points: (N, 3) array of points
        :param color: (N, 3) array of colors
        """
        assert points.shape[1] == 3, "The input array must have shape (N, 3)"
        assert color.shape[1] == 3, "The input array must have shape (N, 3)"
        assert (
            points.shape[0] == color.shape[0]
        ), "The input arrays must have the same length"
        color = (color * 255).astype(np.uint8)
        # Create a structured NumPy array with named fields
        points_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
        )
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = np.array(
            [
                struct.unpack("I", struct.pack("BBBB", *color[i, ::-1], 255))[0]
                for i in range(color.shape[0])
            ]
        )
        # Convert the NumPy array to a PointCloud2 message
        point_cloud = ros_numpy.point_cloud2.array_to_pointcloud2(
            points_arr, rospy.Time.now(), "world"
        )
        self.point_cloud_pub.publish(point_cloud)

    def visualize_circle(self, center, axis, radius,resolution=100):
        marker = Marker()       # Create a marker
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "picking_ring"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # Line width
        # marker.color.a = 1.0  # Don't forget to set the alpha!
        marker.color.a = 0.8  # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        # Generate points on the circle
        for i in range(resolution + 1):
            angle = 2 * math.pi * i / resolution
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2]
            point = Point()
            point.x = x
            point.y = y
            point.z = z
            marker.points.append(point)
        rospy.loginfo(f"Publishing ..... picking ring now")
        self.circle_pub.publish(marker)
    
    #TODO: Need to account axis for draw this picking ring GT
    def visualize_picking_ring_GT(self, resolution=100): 
        marker = Marker()       # Create a marker
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "picking_ring"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # Line width
        marker.color.a = 0.8  # Don't forget to set the alpha!
        marker.color.r = self.picking_ring_color[0]
        marker.color.g = self.picking_ring_color[1]
        marker.color.b = self.picking_ring_color[2]

        # Generate points on the circle
        for i in range(resolution + 1):
            angle = 2 * math.pi * i / resolution
            x = self.fruit_posi_GT[0] + self.vp_sampling_radius * math.cos(angle)
            y = self.fruit_posi_GT[1] + self.vp_sampling_radius * math.sin(angle)
            z = self.fruit_posi_GT[2]
            point = Point()
            point.x = x
            point.y = y
            point.z = z
            marker.points.append(point)
        if self.verbose_rviz:
            rospy.loginfo(f"Publishing picking ring now")
        self.picking_ring_GT_pub.publish(marker)

    def visualize_ROI(self, points: np.array):
        """
        Visualize the ROI voxels in Rviz as a PointCloud2 message
        :param points: (N, 3) array of points
        :param class_ids: (N, 1) array of semantics
        """
        assert points.shape[1] == 3, "The input array must have shape (N, 3)"
        color_roi = struct.unpack("I", struct.pack("BBBB", 0,128,0, 255))[0]       #Green BGR
        
        # Create a structured NumPy array with named fields
        points_dtype = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32),("rgb", np.uint32),])
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = color_roi
        # Convert the NumPy array to a PointCloud2 message
        voxel_points = ros_numpy.point_cloud2.array_to_pointcloud2(points_arr, rospy.Time.now(), self.world_frame_id)
        self.ROI_voxels_pc2_pub.publish(voxel_points)



    def my_visualize_curve(self, x, y, z):
        """
        Visualize the curve in Rviz as a Marker message
        :param x: (N, ) array of x coordinates
        :param y: (N, ) array of y coordinates
        :param z: (N, ) array of z coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "valid_pose"
        marker.id = 0
        # marker.type = Marker.LINE_STRIP
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        # marker.pose.orientation.w = 1.0
        # marker.scale = Vector3(0.02, 0.02, 0.02)
        # marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
        marker.scale.x = 0.5  # Line width
        # marker.color.a = 1.0  # Don't forget to set the alpha!
        marker.color.a = 0.7  # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        for i in range(len(x)):
            marker.points.append(Point(x[i], y[i], z[i]))
        print("Publishing ..... my_visualize_curve now")
        self.curve_pub.publish(marker)
    
    
    def visualize_curve(self, x, y, z):
        """
        Visualize the curve in Rviz as a Marker message
        :param x: (N, ) array of x coordinates
        :param y: (N, ) array of y coordinates
        :param z: (N, ) array of z coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "curve"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.02, 0.02, 0.02)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
        for i in range(len(x)):
            marker.points.append(Point(x[i], y[i], z[i]))
        self.curve_pub.publish(marker)













    def visualize_rois(self, rois: PoseArray) -> None:
        """
        Visualize the ROIs in Rviz as a MarkerArray message
        :param rois: PoseArray of ROIs
        """
        marker_array = MarkerArray()
        for i, pose in enumerate(rois.poses):
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "rois"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale = Vector3(0.09, 0.09, 0.09)
            # marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
            marker_array.markers.append(marker)
        self.rois_pub.publish(marker_array)


    def visualize_camera_bounds(self, bounds: np.array) -> None:
        """
        Visualize the camera bounds in Rviz as a Marker message
        :param bounds: (2, 3) array of camera bounds
        """
        center = np.mean(bounds, axis=0)
        size = np.max(bounds, axis=0) - np.min(bounds, axis=0)
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_bounds"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = Point(*center)
        marker.scale = Vector3(*size)
        # marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.3)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.3)   #Green
        self.camera_bounds_pub.publish(marker)



    def visualize_wm(self, markers: MarkerArray) -> None:
        """
        Visualize the world model in Rviz as a MarkerArray message
        :param markers: MarkerArray of the objects in the world model
        """
        self.world_model_pub.publish(markers)

    def visualize_gt_point_cloud(self, points: np.array, color: np.array) -> None:
        """
        Visualize the point cloud in Rviz as a PointCloud2 message
        :param points: (N, 3) array of points
        :param color: (N, 3) array of colors
        """
        assert points.shape[1] == 3, "The input array must have shape (N, 3)"
        assert color.shape[1] == 3, "The input array must have shape (N, 3)"
        assert (
            points.shape[0] == color.shape[0]
        ), "The input arrays must have the same length"
        color = (color * 255).astype(np.uint8)
        # Create a structured NumPy array with named fields
        points_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ]
        )
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = np.array(
            [
                struct.unpack("I", struct.pack("BBBB", *color[i, ::-1], 255))[0]
                for i in range(color.shape[0])
            ]
        )
        # Convert the NumPy array to a PointCloud2 message
        point_cloud = ros_numpy.point_cloud2.array_to_pointcloud2(
            points_arr, rospy.Time.now(), "world"
        )
        self.point_cloud_pub2.publish(point_cloud)



    def visualize_points(self, points: np.ndarray):
        """
        Visualize the points in Rviz as a Marker message
        :param points: (N, 3) array of point coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.01, 0.01, 0.01)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        for point in points:
            marker.points.append(Point(*point))
        self.points_pub.publish(marker)

    def visualize_pred_points(self, points: np.ndarray):
        """
        Visualize the points in Rviz as a Marker message
        :param points: (N, 3) array of point coordinates
        """
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "pred_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.02, 0.02, 0.02)
        marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.8)
        for point in points:
            marker.points.append(Point(*point))
        self.pred_points_pub.publish(marker)

    def visualize_gain_image(self, image: np.ndarray):
        """
        Visualize the gain image in Rviz as Image message
        :param image: (H, W, 3) array of gain image
        """
        bridge = CvBridge()
        image = (image * 255).astype(np.uint8)[:, :, ::-1]
        image_msg = bridge.cv2_to_imgmsg(image, encoding="passthrough")
        image_msg.header.frame_id = self.camera_frame_id
        image_msg.header.stamp = rospy.Time.now()
        self.gain_image_pub.publish(image_msg)
