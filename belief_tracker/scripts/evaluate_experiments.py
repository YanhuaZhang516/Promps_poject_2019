


import belief_tracker.data_evaluator as data_evaluator

import numpy as np

import matplotlib.pyplot as plt

filename_traj = "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/recorded_data/data_vel_test/tuan/tuan_no_robot.csv"
#filename_traj = "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/recorded_data/data_vel_test/joni/joni_no_robot.csv"
filename_goals = "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/data/goals/default_goals.csv"
evaluator = data_evaluator.DataEvaluator(filename_traj,filename_goals)
_rest_duration_at_goals, dist_total, dist_one_trial,dist_mean, dist_std,time_all,time_one_trial,time_mean,time_std,finished_pieces = evaluator.compute_trajectory_lengths_and_duration()

print "distance total: " + str(dist_total)
print "distance trials: "
print dist_one_trial
print "distance trials mean: " + str(dist_mean)
print "distance trials std: " + str(dist_std)
print "time total: " +str(time_all)
print "time, trials: "
print time_one_trial
print "time trial mean: "+ str(time_mean)
print "time trial std: " +str(time_std)
print "mean restduration at goals "
print _rest_duration_at_goals
print "mean velocity:" + str(dist_total/time_all)
print "finished pieces human: " +str(finished_pieces)


