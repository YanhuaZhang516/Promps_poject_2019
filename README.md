## Intention Aware ProMPs

# Installation Intructions

Clone this repository into the src folder of your ROS/catkin workspace and build the workspace

you might need to install scipy as a requirement

    pip install scipy


# Goal Tracker

The goal tracker learns an incremental Gaussian Mixture Model from zero velocity points

You can start the goal tracker in simulation with

    roslaunch belief_tracker goal_tracker_sim_human.launch 

the goal tracker will not be active unless a start message is published 

from the console you can do this with

    rostopic pub /start_goal_tracker promp_robot_control_msgs/Start "{start: true, robot_file: '', goals_file: '', belief_tracker_parameters_file: '',  belief_tracker_goal_transition_history: 0}"

If you want to see what happens when motions are observed start the csv to tf node to simulate some motion data

    roslaunch belief_tracker csv_to_tf.launch
    
# Belief Tracker

The belief tracker predicts human motions based on the observation history and the goals

you can start the belief tracker in simulation with fixed goals with

    roslaunch belief_tracker belief_tracker_sim_human.launch 

the belief tracker will not be active unless a start message is published 

from the console you can do this with

    rostopic pub /start_belief_trker promp_robot_control_msgs/Start "{start: true, robot_file: '', goals_file: '', belief_tracker_parameters_file: '',  belief_tracker_goal_transition_history: 0}" 

If you want to see what happens when motions are observed start the csv to tf node to simulate some motion data

    roslaunch belief_tracker csv_to_tf.launch




