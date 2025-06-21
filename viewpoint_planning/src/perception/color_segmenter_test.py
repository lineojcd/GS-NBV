import rospy
import cv2
import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

def color_segmentation(color_image: np.array) -> np.array:
    """
    Perform color segmentation on the input image using OpenCV
    :param color_image: input color image
    :return: segmentation mask
    """
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    # Define range of red color in HSV
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([10, 255, 255])
    # Threshold the HSV image to get only white colors
    segmentation_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    print("segmentation_mask is ", segmentation_mask)
    
    
#     segmentation_mask is  [[  0   0   0 ...   0   0   0]
#  [  0   0   0 ...   0   0   0]
#  [  0   0   0 ...   0   0   0]
#  ...
#  [255 255 255 ...   0   0   0]
#  [255 255 255 ...   0   0   0]
#  [255 255 255 ...   0   0   0]]
    
    
    
    return segmentation_mask

rospy.init_node('image_processor', anonymous=True)
image_pub = rospy.Publisher("/segmentation_mask", Image, queue_size=10)
bridge = CvBridge()

# Assuming you have a way to receive or read an image
# color_image = cv2.imread('path_to_your_image.jpg')
color_image = cv2.imread('/home/jcd/Downloads/red_bunny.jpg')

if False:
    mask = color_segmentation(color_image)

    # Convert the mask to a ROS Image message and publish it
    ros_image = bridge.cv2_to_imgmsg(mask, "mono8")
    image_pub.publish(ros_image)
    rospy.spin()


if color_image is not None:
    mask = color_segmentation(color_image)
    
    # Display the original image and the mask side by side
    # Convert mask to BGR for concatenation
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined_image = np.hstack((color_image, mask_bgr))

    cv2.imshow("Original and Segmented Image", combined_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    # Convert the mask to a ROS Image message and publish it
    # ros_image = bridge.cv2_to_imgmsg(mask, "mono8")
    # image_pub.publish(ros_image)
    # rospy.spin()
else:
    print("Image not found. Please check the file path.")