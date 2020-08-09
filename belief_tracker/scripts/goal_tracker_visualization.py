#! /usr/bin/env python

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray


class GoalTrackerVisualization:
    def __init__(self):
        self.max_goal_idx = 0
        self.vel_goal_candidates_markers = MarkerArray()
        self._vel_goal_candidates_publisher = rospy.Publisher('/online_goal_extractor/vel_goal_candidates',
                                                              MarkerArray, queue_size=10)
        self._goals_publisher = rospy.Publisher('/online_goal_extractor/goals', MarkerArray, queue_size=10)


    # def __del__(self):
    #     self._vel_goal_candidates_publisher.unregister()


    def add_vel_goal_candidate(self, point):
        # type: (list) -> None
        marker = Marker()

        marker.header.frame_id = "darias"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "my_namespace"
        marker.id = len(self.vel_goal_candidates_markers.markers) + 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0

        self.vel_goal_candidates_markers.markers.append(marker)

        self._vel_goal_candidates_publisher.publish(self.vel_goal_candidates_markers)


    def update_goal_visualization(self, num_components, goal_means, goal_covs):
        # type: (int, list, list) -> None
        msg = MarkerArray()
        id=0
        for i in range(0, num_components):
            marker = Marker()

            marker.header.frame_id = "darias"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "my_namespace"
            marker.id = id
            id+=1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = goal_means[i][0]
            marker.pose.position.y = goal_means[i][1]
            marker.pose.position.z = goal_means[i][2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = np.sqrt(np.diag(goal_covs[i])[0]) * 2 *2# because this is diameter of markers !!!!!
            marker.scale.y = np.sqrt(np.diag(goal_covs[i])[1]) * 2 *2# because this is diameter of markers !!!!!
            marker.scale.z = np.sqrt(np.diag(goal_covs[i])[2]) * 2 *2# because this is diameter of markers !!!!!
            # print "goal "+str(i)
            # print self.goal_covs
            marker.color.a = 0.5
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 1.0

            msg.markers.append(marker)
            marker = Marker()

            marker.header.frame_id = "darias"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "my_namespace"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = goal_means[i][0]
            marker.pose.position.y = goal_means[i][1]
            marker.pose.position.z = goal_means[i][2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.text = "goal "+ str(i)
            marker.scale.z=0.1
            marker.color.a = 1.
            marker.color.r = 1.
            marker.color.g = 1.0
            marker.color.b = 1.
            marker.id = id
            id += 1

            msg.markers.append(marker)

        self.max_goal_idx = np.max([id, self.max_goal_idx])

        for i in range(id, self.max_goal_idx):
            marker = Marker()

            marker.header.frame_id = "darias"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "my_namespace"
            marker.id = id
            id+=1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = -1.
            marker.pose.position.y = 0.
            marker.pose.position.z = 0.
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            # print "goal "+str(i)
            # print self.goal_covs
            marker.color.a = 0.5
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 1.0

            msg.markers.append(marker)




        self._goals_publisher.publish(msg)
