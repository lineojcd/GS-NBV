#!/usr/bin/env python3
# ROS node to run the viewpoint planning algorithms

import rospy
from viewpoint_planners.gradientnbv_planning import GradientNBVPlanning
from viewpoint_planners.gsnbv_planning import GSNBVPlanning
from viewpoint_planners.scnbv_planning import SCNBVPlanning
from viewpoint_planners.viewpoint_planning import ViewpointPlanning

if __name__ == "__main__":
    rospy.init_node("viewpoint_planning")
    
    # Get the planner argument from the launch file
    planner_choice = rospy.get_param('~planner', 'gsnbv')
    
    # Log which planner is being used
    rospy.loginfo(f"Using planner: {planner_choice}")
    
    if planner_choice == "gsnbv":
        viewpoint_planner = GSNBVPlanning()
    if planner_choice == "scnbv":
        viewpoint_planner = SCNBVPlanning()
    if planner_choice == "gnbv":
        viewpoint_planner = GradientNBVPlanning()
    if planner_choice == "rvp":
        pass
    if planner_choice == "test":
        viewpoint_planner = ViewpointPlanning()
        
    while not rospy.is_shutdown():
        viewpoint_planner.run()
