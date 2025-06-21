#!/usr/bin/env python

import rospy
import tf2_ros
import geometry_msgs.msg
# import tf2_geometry_msgs
import numpy as np
import torch

def get_transform(target_frame, source_frame):
    rospy.init_node('tf_listener_node')

    # Create a TF2 buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Wait for the transform to become available
    try:
        # Wait for up to 5 seconds for the transform
        trans = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(5.0))
        rospy.loginfo("Transform from {} to {}: \n{}".format(source_frame, target_frame, trans))
        return trans
    except tf2_ros.LookupException as e:
        rospy.logerr("Transform not found: {}".format(e))
        return None
    except tf2_ros.ConnectivityException as e:
        rospy.logerr("Transform connectivity error: {}".format(e))
        return None
    except tf2_ros.ExtrapolationException as e:
        rospy.logerr("Transform extrapolation error: {}".format(e))
        return None

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (w, x, y, z)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    # Compute the rotation matrix elements
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                  [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                  [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    return R

def rotation_matrix_to_quaternion(R):
    # Extract the diagonal elements
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([w, x, y, z])


def quaternion_to_matrix_official(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternions: tensor of quaternions (..., 4) ordered (w, x, y, z)
    :return: tensor of rotation matrices (..., 3, 3)
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


if __name__ == '__main__':
    try:
        target_frame = "world"
        source_frame = "j2n6s300_end_effector"
        transform = get_transform(target_frame, source_frame)
        if transform:
            # Do something with the transform
            rospy.loginfo("Translation: x={}, y={}, z={}".format(transform.transform.translation.x,
                                                                 transform.transform.translation.y,
                                                                 transform.transform.translation.z))
            rospy.loginfo("Rotation: x={}, y={}, z={}, w={}".format(transform.transform.rotation.x,
                                                                    transform.transform.rotation.y,
                                                                    transform.transform.rotation.z,
                                                                    transform.transform.rotation.w))
    except rospy.ROSInterruptException:
        pass
    
    
    exit()
    # Example usage:
    q1 = (-0.037, -0.026, -0.830, 0.556)  # T_w_c Q
    q2 = (0.698, 0.116, 0.697, 0.116)    # T_c_ee Q

    q_result = quaternion_multiply(q1, q2) # T_w_ee Q
    print(q_result)

    
    Rq1 = quaternion_to_rotation_matrix(q1)  # T_w_c Q
    print("my Rq1:", Rq1)
    print(rotation_matrix_to_quaternion(Rq1))
    Rq2 = quaternion_to_rotation_matrix(q2)    # T_c_ee Q
    print(rotation_matrix_to_quaternion(Rq2))

    R_result = Rq1 * Rq2  # T_w_ee Q
    q_result = rotation_matrix_to_quaternion(R_result)
    print(q_result)
    
    # Define the quaternion
    q1_ = [-0.037, -0.026, -0.830, 0.556] # T_w_c Q
    q2_ = [0.698, 0.116, 0.697, 0.116]    # T_c_ee Q
    q3_ = [0.462, -0.491, 0.305, 0.672]    # T_w_ee Q
    # Convert to a TensorFlow tensor
    q1_tensor = torch.tensor(q1_)
    q2_tensor = torch.tensor(q2_)
    q3_tensor = torch.tensor(q3_)
    Rq1 = quaternion_to_matrix_official(q1_tensor)  # T_w_c Q
    print("let's check here:")
    print(Rq1)
    Rq2 = quaternion_to_matrix_official(q2_tensor)  # T_c_ee Q
    print(Rq2)
    R_result =  Rq1  * Rq2   # T_w_ee Q
    print(q_result)
    q_result = rotation_matrix_to_quaternion(R_result)
    print(R_result)
    Rq3 = quaternion_to_matrix_official(q3_tensor)  # T_c_ee Q
    print(Rq3)
    
    
    Rw_c = torch.tensor([[-0.3790403,  0.9248091,  0.0325054], 
                         [-0.9209614, -0.3804262,  0.0842972],
                         [0.0903247,  0.0020158,  0.9959103]
                         ], dtype=torch.float32)
    Rc_ee = torch.tensor([[0.0013951,  0.0002320,  0.9999990], 
                         [0.3236643, -0.9461719, -0.0002320],
                         [0.9461710,  0.3236643, -0.0013951]
                         ], dtype=torch.float32)
    C = torch.matmul(Rw_c, Rc_ee)
    print("Result using PyTorch:")
    print(C)
