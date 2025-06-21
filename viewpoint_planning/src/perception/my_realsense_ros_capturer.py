import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

class MyRealsenseROSCapturer:
    """
    Gets the color and depth frames of a realsense RGB-D camera from ROS (D4XX, D5XX).
    """
    def __init__(self):
        # Color and depth frames.
        self.color_image = None
        self.depth_image = None
        
        # Realsense need some spesific setting to get this topic
        self.points = None
        self.camera_info = None
        self.use_sim = rospy.get_param("use_sim", True)
        self.camera_type = rospy.get_param("camera_type", "d435")
        self.camera_depth_unit = rospy.get_param("camera_depth_unit", "1000.0")
        
        # Realsense ROS topics.
        if self.camera_type == "d435":                                  # MyCamera: Realsense D435i
            rospy.Subscriber("/d435/color/image_raw", Image, self.color_callback, queue_size=2)
            rospy.Subscriber("/d435/color/camera_info",CameraInfo,self.info_callback,queue_size=1)
            rospy.Subscriber("/d435/depth/image_raw",Image,self.depth_callback,queue_size=1)
        else:
            rospy.Subscriber("/camera/color/image_rect_color", Image, self.color_callback, queue_size=2)
            rospy.Subscriber("/camera/aligned_depth_to_color/camera_info",CameraInfo,self.info_callback,queue_size=1)
            rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",Image,self.depth_callback,queue_size=1)
            # Realsense need some spesific setting to get this topic
            rospy.Subscriber("/camera/depth_registered/points",PointCloud2,self.points_callback,queue_size=1,buff_size=100000000)
            
        # Publish annotated_image by Yolo for visualization
        self.image_pub = rospy.Publisher("/output/annotated_image", Image, queue_size=2)

    def color_callback(self, msg):
        # Subscriber callback for color image.
        data = ros_numpy.numpify(msg)
        self.color_image = data[:, :, ::-1]

    def depth_callback(self, msg):
        # Subscriber callback for depth image.
        data = ros_numpy.numpify(msg)
        if  self.use_sim:
            data = data.astype("float32") / self.camera_depth_unit
        self.depth_image = data
        # # Calculate the minimum and maximum values
        # print("Raw depth data: min ", np.nanmin(self.depth_image)," max ", np.nanmax(self.depth_image))

    def points_callback(self, msg): 
        # Subscriber callback for point cloud.  for Realsense L515 not for D435
        data = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False)
        self.points = np.reshape(data, (msg.height, msg.width, 3))
        if self.use_sim:
            # because the initial Z-coords of arm is 0.024
            self.points[..., 1] += 0.024                    # I can change it here if I use L515

    def info_callback(self, msg):
        # Subscriber callback for camera info.
        self.camera_info = msg

    def get_frames(self):
        # Get the next color and depth frame
        color_output = {}
        depth_output = {}
        color_output["color_image"] = self.color_image
        depth_output["depth_image"] = self.depth_image
        depth_output["points"] = self.points
        return self.camera_info, color_output, depth_output
