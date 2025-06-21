from cv_bridge import CvBridge

# Test Case
# Test with a dummy numpy array to ROS Image


bridge = CvBridge()
import numpy as np
dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
ros_image = bridge.cv2_to_imgmsg(dummy_image, "bgr8")
