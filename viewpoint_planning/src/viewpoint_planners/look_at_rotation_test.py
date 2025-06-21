import rospy
import numpy as np
import tf2_ros

from geometry_msgs.msg import Point, Quaternion, Pose, PoseArray, PoseStamped
from scipy.spatial.transform import Rotation as scipy_r
from pytransform3d.transformations import transform_from_pq, pq_from_transform
from scipy.spatial.transform import Rotation as R


def axangle2quat(vector: np.array, theta: float, is_normalized=False):
    if not is_normalized:
        vector = vector / np.linalg.norm(vector)
    t2 = theta / 2.0
    st2 = np.sin(t2)
    return np.concatenate(([np.cos(t2)], vector * st2))

def look_at_rotation(
    eye: np.array, target: np.array, ref: np.array = [-1.0, 0.0, 0.0]
) -> np.array:
    dir = target - eye
    dir = dir / np.linalg.norm(dir)

    rot_axis = np.cross(ref, dir)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(ref, dir))

    if np.isnan(rot_axis).any():
        return np.array([1.0, 0.0, 0.0, 0.0])

    quat = axangle2quat(rot_axis, rot_angle, True)
    return quat

if __name__ == '__main__':
    init_target_distance = 0.55 
    init_tar_posi = np.array([0.5, -0.4, 1.18-0.7])
    posi = np.array([init_tar_posi[0], init_tar_posi[1] + init_target_distance, init_tar_posi[2]])
    orientation = look_at_rotation(posi, init_tar_posi)
    camera_pose = np.concatenate((posi, orientation)) 
    print("camera_pose is :", camera_pose)
    