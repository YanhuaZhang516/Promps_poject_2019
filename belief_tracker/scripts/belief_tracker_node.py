#!/usr/bin/python

# System Imports

import rospy
import numpy as np

import  belief_tracker.belief_tracker_class as belief_tracker_class




if __name__ == '__main__':
    rospy.init_node("belief_tracker_node")
    belief_tracker = belief_tracker_class.BeliefTracker()

    # TODO: Investigate, why "Rate" is not working :(
    frequency = rospy.get_param("frequency", 30.0)

    rate = rospy.Rate(frequency)
    rospy.spin()
    # while not rospy.is_shutdown():
    #     belief_tracker.tick()
    #     rate.sleep()
