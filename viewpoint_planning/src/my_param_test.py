#!/usr/bin/env python3
import rospy

def example_node():
    rospy.init_node('example_node')

    # Retrieve the model_init_pose parameter
    model_init_pose = rospy.get_param('model_init_pose')
    # Split the string and convert each element to a float
    model_init_pose = [float(x) for x in model_init_pose.split()]

    rospy.loginfo(f"Model initial pose: {model_init_pose}")
    rospy.loginfo(f"Type of model_init_pose: {type(model_init_pose)}")
    for i in model_init_pose:
        print(type(i))
        print(i)

if __name__ == "__main__":
    example_node()
