#!/usr/bin/python

# System Imports
import tf
import csv
import time
import rospy
import numpy as np

from scipy.stats import multivariate_normal

import pickle

# ROS Messages
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker

# Custom Messages
from promp_robot_control_msgs.msg import Start
from promp_robot_control_msgs.srv import Goals, GoalsResponse
from promp_robot_control_msgs.msg import PredictedTrajectories

# Custom Imports
from history import *
import rospkg
from numpy import genfromtxt

def normalize(v):
    """
    Normalizes a vector.
    Parameters
    ----------
    v : np.ndarray
        Vector to normalize

    Returns
    -------
    np.ndarray :
        Normalized vector.

    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm






class BeliefTracker:
    def __init__(self, use_ros = True):
        # set this to false to use it for just testing with csv files e.g.
        self.use_ros=use_ros
        # ROS Essentials
        self.pkg_path = rospkg.RosPack().get_path('belief_tracker')

        self._pub_belief = rospy.Publisher("/belief", Float64MultiArray, queue_size=1)
        self._pub_curr_goal = rospy.Publisher('/curr_goal', PoseStamped, queue_size=1)
        self._pub_human_pos = rospy.Publisher('/human_pos', PoseStamped, queue_size=1)
        self._pub_sampled_trajectories = rospy.Publisher("/sampled_trajectories", PredictedTrajectories, queue_size=1)

        # Visualization Publishers
        self._pub_trajectory_viz = rospy.Publisher("/sampled_trajectories_viz", MarkerArray, queue_size=1)
        self._pub_curr_dir = rospy.Publisher('/curr_direction', Marker, queue_size=1)
        self._pub_belief_vis = rospy.Publisher('/goal_belief_vis', MarkerArray, queue_size=1)

        self.goal_tracker_call = rospy.ServiceProxy("/gt_get_goals", Goals)
        self._tf_listener = tf.TransformListener()

        self.importance_of_prior_in_belief_update = 10

        self.current_init_belief=[]

        # Belief Tracker Essentials
        self._darias_frame = "/darias"
        self._human_frame = rospy.get_param("human_frame", "/human")
        self._frequency = rospy.get_param("frequency", 10.0)
        print self._frequency
        self._belief_tracker_only = rospy.get_param("belief_tracker_only", False)

        self.initialize_parameters()

        goal_change_probabilities_filename = self.pkg_path + "/data/transition_probabilities/test_prob.csv"
        self._goal_change_prob = genfromtxt(goal_change_probabilities_filename, delimiter=',')
        print self._goal_change_prob

        self.start_belief_sub = rospy.Subscriber("/start_belief_tracker", Start, self._start_callback)

        rospy.logwarn("[BeliefTracker] Waiting for Start Message ...")

    def initialize_parameters(self):

        self.update_factor_transition_probabilities_online = True
        self.update_factor_transition_probabilities = 0.1
        self.goals_reached_history =[]

        self._estimated_positions = []
        self._current_human_pos = []
        self._current_belief = []
        self._direction = []
        self._goals = []
        self._velocities = []
        self._last_belief_over_history = []


        self._reached_goals = []
        self.threshold_goal_reached = 0.15  # TODO this should be consistent with data eval parameters instead of hardcoded

        self._personalized_vel = 0.6
        self._resting_duration_at_goals = []

        self._current_velocity = 0.0
        self._num_goals = 0
        self._human_std_dev = 0.01
        self._obs_variance = 0.002#0.0005


        self._last_time = time.time()
        self._history = History()
        self._is_belief_tracker_ready = False

        self._max_belief_history = int(self._frequency * 0.6) #1.0
        self._max_vel_history = int(self._frequency * 0.3)# *1.0
        self._max_velocity = 1.0
        self._belief_threshold = 0.8
        self._belief_threshold_min = 0.3

        self._num_sampled_trajectories = 20
        #self._t_predict = int(self._frequency * 2.0)
        print
        self._t_predict_dt = 0.3#(1.0 / self._frequency)*6
        print "dt predict "+str(self._t_predict_dt)
        self._t_predict = int(2.0 / self._t_predict_dt) #2.5 vorher
        print "T predict " + str(self._t_predict)
        print "t_pred:"+str(self._t_predict)


    def _init_goals_from_csv(self, goals_file_path):
        """
       Collects the goals from the given CSV file path.

       Parameters
       ==========
       goals_file_path : str
            Path to the CSV goals file.
       """
        data_array = np.genfromtxt(goals_file_path, dtype=float, delimiter=",", skip_header=1)[:, 0:3]
        for i in range(data_array.shape[0]):
            self._goals.append(data_array[i])
        #
        # with open(goals_file_path) as csv_file:
        #     reader = csv.reader(csv_file, delimiter=',')
        #     line_count = 0
        #     for row in reader:
        #         if line_count > 0:
        #             goal = []
        #             for point in row:
        #                 goal.append(float(point))
        #             self._goals.append(np.array(goal))
        #         line_count += 1
        self._num_goals = len(self._goals)
        rospy.logwarn("[BeliefTracker] Read {} goals from file {} !".format(self._num_goals, goals_file_path))

    def _init_belief(self):
        """Initializes the belief uniformly."""
        belief = []
        for i in range(self._num_goals):
            belief.append(1.0 / self._num_goals)
        self._last_belief_over_history = np.copy(belief)
        return belief

    def _is_human_in_range_of_goal(self, current_position):
        # type: (np.ndarray) -> bool
        for g in range(self._num_goals):
            if np.linalg.norm(self._goals[g] - current_position) <= self._radius_around_goal[g]:
                return [True, g]

        return [False, -1]

    def _update_goals(self):
        """
        Updates the goals by calling the GoalTracker and requesting the new goals.
        The number of goals and the current belief will then be initialized again.
        """
        print"updating goals"
        response = self.goal_tracker_call()  # type: GoalsResponse
        self._goals = []
        for goal in response.goals:  # type: Point
            self._goals.append([goal.x, goal.y, goal.z])
        self._num_goals = len(self._goals)

        self._current_belief = self._init_belief()

    def transition_human(self, position, velocity, goal, dt):
        # type: (np.ndarray, float, np.ndarray, float) -> np.ndarray
        """
        The human model to compute the next position given the current position,
        the velocity and the goal. Returns a biased position vector.
        :param position: Current position of the human
        :param velocity: Current velocity of the human.
        :param goal: Goal position of the human.
        :param dt: Time difference.
        :return: New position.
        """

        std_noise=np.copy(self._human_std_dev)
        if velocity <= 0.2:
            std_noise = std_noise*velocity

        noise = np.random.normal(loc=0.0, scale=std_noise, size=3)
        change_in_position = (velocity * dt * normalize(goal - position))
        dist_to_goal=np.linalg.norm(goal-position)
        dist_change = np.linalg.norm(change_in_position)
        if dist_change>dist_to_goal:
            change_in_position = goal - position

        change_in_position+= noise
        return position + change_in_position

    def compute_observation_likelihood(self, current_observation, last_observation, goal, velocity, dt):
        # type: (np.ndarray, np.ndarray, np.ndarray, float, float) -> (float, np.ndarray)
        """Computes the likelihood for the current observation.
        :param current_observation: Current observation
        :param last_observation: Last observation
        :param goal: Goal that the human is heading towards to
        :param velocity: Velocity at which the human moves
        :param dt: Time step
        :return: Likelihood
        """
        calculated_position = self.transition_human(last_observation, velocity, goal, dt)
        likelihood = multivariate_normal.pdf(x=current_observation, mean=calculated_position,
                                             cov=(self._obs_variance * np.identity(3)))
        return likelihood, calculated_position

    def update_belief_close_to_goal(self, current_position, current_belief):
        [is_at_goal, g] = self._is_human_in_range_of_goal(current_position)

        #print " at "+str(g)
        likelihoods = np.zeros(self._num_goals)+0.001
        likelihoods[g] = 1.0
        new_belief = []
        normalization_factor = 0.0

        for goal_idx in range(self._num_goals):
            normalization_factor += likelihoods[goal_idx] * current_belief[goal_idx]

        for goal_idx in range(self._num_goals):
            if normalization_factor==0.0:
                print "norm : " +str(normalization_factor)
                print likelihoods
                print current_belief
            prob = (likelihoods[goal_idx] * current_belief[goal_idx]) / normalization_factor
            new_belief.append(prob)

        return new_belief


    def update_belief_once(self, current_observation, last_observation, avg_vel, dt, current_belief):
        # type: (np.ndarray, np.ndarray, float, float, list) -> (list, list)
        """Updates the belief one time.
        :param current_observation: Current observation
        :param last_observation: Last observation
        :param avg_vel: Average velocity
        :param dt: Time step
        :param current_belief: Current Belief
        :return: Updated belief and estimated positions
        """



        new_belief = []
        likelihoods = []
        estimated_positions = []
        normalization_factor = 0.0

        # Compute the likelihoods
        for goal_idx in range(self._num_goals):
            obs_likelihood, calculated_position = self.compute_observation_likelihood(current_observation,
                                                                                      last_observation,
                                                                                      self._goals[goal_idx],
                                                                                      avg_vel, dt)
            estimated_positions.append(calculated_position)
            obs_likelihood += 1
            likelihoods.append(obs_likelihood)
            normalization_factor += obs_likelihood * current_belief[goal_idx]




        #for i in range(self.importance_of_prior_in_belief_update):
        #normalization_factor = 0.0
        #tmp_belief = []
        # Compute new belief
        for goal_idx in range(self._num_goals):
            prob = (likelihoods[goal_idx] * current_belief[goal_idx])/normalization_factor

            new_belief.append(prob)

        #tmp_belief = np.array(tmp_belief) / normalization_factor


        #new_belief = tmp_belief
        return [new_belief, estimated_positions]

    def _compute_velocity(self, last_pos, current_pos, dt):
        # type: (np.ndarray, np.ndarray, float) -> (float, float)
        """
        Computes the current velocity, as well as the average velocity over the history.
        :param last_pos:
        :param current_pos:
        :param dt: Time step
        :return: Current velocity and average velocity
        """
        diff = current_pos - last_pos
        raw_vel = np.sqrt(np.dot(diff, diff)) / dt
        if raw_vel > self._max_velocity:
            raw_vel = self._max_velocity - np.random.normal(0, 0.01)
        # avg_vel = (np.sum(self._history.velocities) + raw_vel) / (self._history.size() + 1)
        self._velocities.append(raw_vel)
        avg_vel = float(np.mean(self._velocities[-self._max_vel_history:]))

        return float(raw_vel), float(avg_vel)

    def _update_belief_over_history(self):
        # type: () -> list
        """
        Updates the belief over the history (last n entries).
        History has to have at least 2 entries!
        The current observation will not be included here!
        :return: The new belief
        """

        if len(self.current_init_belief)==0:
            #print"init normal:"
            belief = self._init_belief()
        else:
            #print"init transitions:"
            #print self.current_init_belief
            belief = np.copy(self.current_init_belief)
            self._last_belief_over_history = np.copy(belief)

        #print "after"
        if self._history.size() < self._max_belief_history:
            start_idx_history_window = 1
        else:
            start_idx_history_window = self._history.size() - self._max_belief_history

        for t in range(start_idx_history_window, self._history.size()):
            dt = self._history.dts[t]
            current_obs = self._history.observations[t]
            last_obs = self._history.observations[t - 1]

            vel, avg_vel = self._compute_velocity(last_obs, current_obs, dt)

            # TODO: Belief will be computed multiple times here too!
            belief, _ = self.update_belief_once(current_obs, last_obs, avg_vel, dt, belief)

        return belief

    def tick(self, msg):

        """
        If the Belief Tracker got the Start message, this method will be run outside of the Belief Tracker at a
        specified rate and will compute the Belief continuously.
        """

        if self._is_belief_tracker_ready:
            start = time.time()
            self._compute_belief()
            dur = (time.time() -start)
            if dur < 1.0/self._frequency:
                rospy.sleep((1.0/self._frequency) - dur)
        else:
            print"too slow !!!!!!!"


    # ==================================================================================================================
    # ROS Callbacks
    # ==================================================================================================================

    def _start_callback(self, msg):
        """
        Starts the Belief Tracker and initializes the goals and the belief.

        Parameters
        ----------
        msg : Start
            Start message containing the file for initial goals.

        """

        self.initialize_parameters() # think if we want this

        if not msg.goals_file:
            
            rospy.logerr("[BeliefTracker] Goals File Path is empty! Using default path for the CSV file!")
            self._init_goals_from_csv(self.pkg_path+'/data/goals.csv')
            #exit(1)



        # Initialize goals and belief
        else:
            self._init_goals_from_csv(msg.goals_file)
        
        self._current_belief = self._init_belief()
        self._radius_around_goal = [0.1 for _ in range(self._num_goals)]

        # Allow the Belief computation loop to start
        self._is_belief_tracker_ready = True

        rospy.loginfo("[BeliefTracker] Belief Tracker started!")

        parameter_file = msg.belief_tracker_parameters_file

        if not parameter_file:
            parameter_file = self.pkg_path+"/data/transition_probabilities/belief_tracker_params_default.pkl"

        transition_history_length = msg.belief_tracker_goal_transition_history
        if transition_history_length == 0:
            print "transition history length not specified, setting it per default to 1"
            transition_history_length = 1

        self.update_belief_tracker_parameters(parameter_file,transition_history_length)

        self._reached_goals = []
        self._timer = rospy.Timer(rospy.Duration(1.0 / self._frequency), self.tick)

    def update_belief_tracker_parameters(self,filename, belief_tracker_goal_transition_history):

        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

            transition_probabilities_history_vector = data["transition_probabilities_history_vector"]
            _goal_change_prob_all = data["transition_probabilities"]
            resting_duration = data["resting_duration"]


        idx_history_length = np.where(transition_probabilities_history_vector == belief_tracker_goal_transition_history)[0]

        if len(idx_history_length) == 1:
            self.transition_probabilities_history_vector = transition_probabilities_history_vector
            self._goal_change_prob_all = _goal_change_prob_all
            self.idx_history_change_probs = idx_history_length[0]
            self._goal_change_prob = self._goal_change_prob_all[idx_history_length[0]]
            self._goal_change_prob_length = belief_tracker_goal_transition_history

            self._resting_duration_at_goals = np.array([0.,0.0,0.,0.])#resting_duration
            print resting_duration
            self.curr_stay_duration_goals = np.copy(self._resting_duration_at_goals)

        else:
            print "requested history length not or multiple times in specified history vector not updating"

    def update_transition_probabilities(self):
        goals = self._reached_goals
        print "reached goals"
        print goals
        print "all 0:"
        #print self._goal_change_prob_all[0]

        for i in range(len(self._goal_change_prob_all)):
            #print "dims"
            #print self._goal_change_prob_all[i].ndim
            if self._goal_change_prob_all[i].ndim <= len(self._reached_goals):
                indices = self._reached_goals[
                          len(self._reached_goals) - self._goal_change_prob_all[i].ndim:len(self._reached_goals)]
                indices_small = self._reached_goals[
                          len(self._reached_goals) - self._goal_change_prob_all[i].ndim:len(self._reached_goals)-1]
                first_time = False
                for g in range(self._num_goals):
                    idx_tmp = indices_small[:]
                    idx_tmp.append(g)
                    if self._goal_change_prob_all[i][tuple(idx_tmp)]<0.0:
                        first_time=True
                        self._goal_change_prob_all[i][tuple(idx_tmp)] = 0.0
                        print "updating -1 probabilitiy"

                    self._goal_change_prob_all[i][tuple(idx_tmp)]*=  (1.-self.update_factor_transition_probabilities)

                # if self._goal_change_prob_all[i].ndim == 2:
                #     print "idx:"
                #     print indices
                #
                #     self._goal_change_prob_all[i][: , self._reached_goals[-1]] *=  (1.-self.update_factor_transition_probabilities)
                # elif self._goal_change_prob_all[i].ndim == 3:
                #     self._goal_change_prob_all[i][:,:, self._reached_goals[-1]] *= (1. - self.update_factor_transition_probabilities)
                # elif self._goal_change_prob_all[i].ndim == 4:
                #     self._goal_change_prob_all[i][:,:,:, self._reached_goals[-1]] *= (1. - self.update_factor_transition_probabilities)
                # elif self._goal_change_prob_all[i].ndim == 5:
                #     self._goal_change_prob_all[i][:,:,:,:, self._reached_goals[-1]] *= (1. - self.update_factor_transition_probabilities)
                #print "dims"
                #self._goal_change_prob_all[i].ndim

                # print "idx:"
                # print indices
                if i ==3:
                    print "idx:"
                    print indices
                    print "idx_small:"
                    print indices_small
                if first_time:
                    self._goal_change_prob_all[i][tuple(indices)] = 1.0
                else:
                    self._goal_change_prob_all[i][tuple(indices)] += 1.0*self.update_factor_transition_probabilities

        print (self._goal_change_prob_all[0].shape)

        #for i in range(self._num_goals):

        self._goal_change_prob = self._goal_change_prob_all[self.idx_history_change_probs]
        #print self._goal_change_prob
        # print"0|1"
        # print self._goal_change_prob[1,0]
        # print"0|3"
        # print self._goal_change_prob[ 3,0]

        print "p(3|1,0,2,0):  " + str( self._goal_change_prob_all[3][1,0,2,0,3])

        print "p(0|0,2,0,3):  " + str(self._goal_change_prob_all[3][0, 2, 0, 3, 0])
        print "p(1|0,2,0,3):  " + str(self._goal_change_prob_all[3][0,2,0,3,1])
        print "p(2|0,2,0,3):  " + str(self._goal_change_prob_all[3][0, 2, 0, 3, 2])

        print "p(0|0,1,0,3):  " + str(self._goal_change_prob_all[3][0, 1, 0, 3, 0])
        print "p(1|0,1,0,3):  " + str(self._goal_change_prob_all[3][0,1,0,3, 1])
        print "p(2|0,1,0,3):  " + str(self._goal_change_prob_all[3][0,1, 0, 3, 2])

        print "p(2|0,3):  " + str(self._goal_change_prob_all[1][0, 3, 2])
        print "p(1|0,3):  " + str(self._goal_change_prob_all[1][0, 3, 1])
        print "p(0|0,3):  " + str(self._goal_change_prob_all[1][0, 3, 0])

        print "p(2|1):  " + str(self._goal_change_prob_all[1][1,2])
        print "p(0|1):  " + str(self._goal_change_prob_all[1][1,0])
        print "p(3|1):  " + str(self._goal_change_prob_all[1][1,3])
        #print "p(0|1):  " + str(self._goal_change_prob_all[3][1, 0])
        #print "p(0|1):  " + str(self._goal_change_prob_all[3][1, 0])
        #print "p(0|1):  " + str(self._goal_change_prob_all[3][1, 0])
        #print "p(0|1):  " + str(self._goal_change_prob_all[3][1, 0])

    def sim_belief_from_csv_file(self,filename,start,stop,step,goals_file,parameter_file):


        with open(filename, "r") as csv_file:
            trajectory = []
            times=[]
            beliefs=[]

            csv_reader = csv.DictReader(csv_file, delimiter=",")
            for row in csv_reader:
                trajectory.append([float(row["human_x"]), float(row["human_y"]), float(row["human_z"])])
                times.append(float(row['time_s']))



        duration = times[-1]-times[0]

        self._frequency = (len(times)/step)/duration

        self.initialize_parameters()  # think if we want this

        # Initialize goals and belief
        self._init_goals_from_csv(goals_file)
        self._current_belief = self._init_belief()
        self._radius_around_goal = [0.1 for _ in range(self._num_goals)]

        # Allow the Belief computation loop to start
        self._is_belief_tracker_ready = True

        transition_history_length = 4

        self.update_belief_tracker_parameters(parameter_file, transition_history_length)

        self._reached_goals = []


        sampled_traj=[]
        avg_vels = []
        at_goals=[]
        stay_durs=[]
        for i in range(start,np.min([stop,len(trajectory)]),step):
            current_human_pos = np.asarray(trajectory[i])
            current_time = times[i]
            preds, at_goal, avg_vel=self._compute_belief_from_pose_and_time(current_human_pos, current_time)
            sampled_traj.append(preds)
            avg_vels.append(avg_vel)
            at_goals.append(at_goal)
            beliefs.append(np.copy(self._current_belief))
            stay_durs.append(np.copy(self.curr_stay_duration_goals))
        return sampled_traj,beliefs,avg_vels,at_goals, self._radius_around_goal, self._max_vel_history,stay_durs



    def at_goal(self, position, goal, threshold):
        # type: (np.ndarray, np.ndarray, float) -> bool
        return np.linalg.norm(goal - position) <= threshold




    def _compute_belief(self):
        """
        Computes the belief for the current environmental state.
        """
        # Compute current dt
        current_time = time.time()





        # Get the current human position
        try:
            (current_human_pos, rotation) = self._tf_listener.lookupTransform(self._darias_frame, self._human_frame,
                                                                              rospy.Time(0))
            current_human_pos = np.asarray(current_human_pos)

        except (tf.ExtrapolationException, tf.ConnectivityException, tf.LookupException):
            return

        self._compute_belief_from_pose_and_time(current_human_pos, current_time)



    def _compute_belief_from_pose_and_time(self, current_human_pos, current_time):
        self._current_human_pos = current_human_pos
        # Update the Goals by calling the GoalTracker if goal tracker is active.
        if not self._belief_tracker_only:
            self._update_goals()


        dt = current_time - self._last_time

        # Store history over reached goals for better transition probability predictions

        for g_idx in range(len(self._goals )):

            goal = np.asarray(self._goals[g_idx])


            if self.at_goal(current_human_pos, goal, self.threshold_goal_reached):
                if len(self._reached_goals)>0:

                    if not g_idx==self._reached_goals[-1]:


                        self._reached_goals.append(g_idx)
                        probs,_ = self.compute_transition_probs_for_goal_from_history(np.copy(self._reached_goals),g_idx)

                        #self._current_belief = list(np.array(probs)+0.001)

                        #self._history = History()
                        print "recomputing prior"
                        print self._current_belief
                        self.current_init_belief = list(np.copy(probs)+0.001)
                        if self.update_factor_transition_probabilities_online:
                            self.update_transition_probabilities()

                else:

                    self._reached_goals.append(g_idx)
                    probs,_ = self.compute_transition_probs_for_goal_from_history(list(np.copy(self._reached_goals)), np.copy(g_idx))
                    #self._current_belief = list(np.array(probs)+0.001)
                    #self._history = History()
                    print "recomputing prior"
                    print self._current_belief
                    self.current_init_belief = list(np.copy(probs)+0.001)
                    if self.update_factor_transition_probabilities_online:
                        self.update_transition_probabilities()

        # print"reached goals:"
        # print self._reached_goals





        # This is the first observation, just fill up the history
        if self._history.size() < 1:

            # if len(self.current_init_belief) == 0:
            #     print"init normal:"
            #     self._current_belief = self._init_belief()
            # else:
            #     print"init transitions:"
            #     print self.current_init_belief
            #     self._current_belief = self.current_init_belief

            last_obs = current_human_pos
            vel, avg_vel = self._compute_velocity(last_obs, current_human_pos, dt)
            self._current_belief, self._estimated_positions = self.update_belief_once(current_human_pos, last_obs,
                                                                                      avg_vel, dt,
                                                                                      self._current_belief)

            self._last_time = current_time
            self._history.add_entry(current_human_pos, vel, dt)
            self._direction = self.compute_direction(current_human_pos)

            belief_msg = Float64MultiArray()
            belief_msg.data = self._current_belief
            if self.use_ros:
                self._pub_belief.publish(belief_msg)


            return [],[],avg_vel

        elif self._history.size() == 1:
            last_obs = self._history.last_entry()[0]
            (vel, avg_vel) = self._compute_velocity(last_obs, current_human_pos, dt)
            self._current_belief, self._estimated_positions = self.update_belief_once(current_human_pos, last_obs,
                                                                                      avg_vel, dt,
                                                                                      self._current_belief)
            self._last_time = current_time
            self._history.add_entry(current_human_pos, vel, dt)
            self._direction = self.compute_direction(current_human_pos)

            belief_msg = Float64MultiArray()
            belief_msg.data = self._current_belief
            if self.use_ros:
                self._pub_belief.publish(belief_msg)
            return [],[],avg_vel

        # Update the belief with the current observation
        last_obs = self._history.last_entry()[0]
        vel, avg_vel = self._compute_velocity(last_obs, current_human_pos, dt)

        # if avg_vel>=0.21:
        #     print "set velocity to 0.6"
        #     avg_vel=0.6
        # else:
        #     print "not"

        self._current_velocity = np.copy(avg_vel)

        #print(avg_vel)

        [is_at_goal, goal_index] = self._is_human_in_range_of_goal(np.asarray(current_human_pos))
        if not is_at_goal:
            self.curr_stay_duration_goals = np.copy(self._resting_duration_at_goals)
            # Only update the belief, if human is moving
            if avg_vel > 0.2:
                # First, compute the belief over the history (without the current observation)
                belief_over_history = self._update_belief_over_history()

                new_belief, self._estimated_positions = self.update_belief_once(current_human_pos, last_obs, avg_vel, dt,
                                                                                belief_over_history)
                sampled_trajectories = self._generate_trajectories(vel, self._t_predict_dt, self._num_sampled_trajectories, self._t_predict,
                                                                   new_belief)
                self._current_belief = np.copy(new_belief)
                self._last_time = current_time
                self._direction = self.compute_direction(current_human_pos)
                self._last_belief_over_history = belief_over_history

                belief_msg = Float64MultiArray()
                belief_msg.data = self._current_belief
                if self.use_ros:
                    self._pub_belief.publish(belief_msg)
            else:
                # TODO: Set probability to 1 ,if near a goal?
                # Generate trajectories for prediction
                _, self._estimated_positions = self.update_belief_once(current_human_pos, last_obs, avg_vel, dt,
                                                                       self._current_belief)
                sampled_trajectories = self._generate_trajectories(vel, self._t_predict_dt, self._num_sampled_trajectories, self._t_predict,
                                                                   self._current_belief)
                self._last_time = current_time
                self._direction = self.compute_direction(current_human_pos)
                belief_msg = Float64MultiArray()
                belief_msg.data = self._current_belief
                if self.use_ros:
                    self._pub_belief.publish(belief_msg)
        # At a goal
        else:

            if avg_vel > 0.2:
                #
                # if self.leaving_goal_first_time:
                #     self.leaving_goal_first_time = False
                #
                #     print "recompute priors"

                # TODO: Set probability to 1 ,if near a goal?
                # Generate trajectories for prediction
                _, self._estimated_positions = self.update_belief_once(current_human_pos, last_obs, avg_vel, dt,
                                                                       self._current_belief)
                sampled_trajectories = self._generate_trajectories(self._personalized_vel, self._t_predict_dt, self._num_sampled_trajectories, self._t_predict,
                                                                   self._current_belief)
                self._last_time = current_time
                self._direction = self.compute_direction(current_human_pos)
                belief_msg = Float64MultiArray()
                belief_msg.data = self._current_belief
                if self.use_ros:
                    self._pub_belief.publish(belief_msg)
            else:
                # self.leaving_goal_first_time _update_belief_over_history= True
                self.curr_stay_duration_goals[goal_index] =  np.max([0.,self.curr_stay_duration_goals[goal_index] - 1./self._frequency])
                # First, compute the belief over the history (without the current observation)
                belief_over_history = self._update_belief_over_history()

                # TODO: Maybe remove here
                self._estimated_positions = [current_human_pos for _ in range(self._num_goals)]

                new_belief = self.update_belief_close_to_goal(current_human_pos, self._current_belief)
                sampled_trajectories = self._generate_trajectories(self._personalized_vel, self._t_predict_dt, self._num_sampled_trajectories,
                                                                   self._t_predict,
                                                                   new_belief)
                self._current_belief = np.copy(new_belief)
                self._last_time = current_time
                self._direction = self.compute_direction(current_human_pos)
                self._last_belief_over_history = belief_over_history

                belief_msg = Float64MultiArray()
                belief_msg.data = self._current_belief
                if self.use_ros:
                    self._pub_belief.publish(belief_msg)

        # Add the current observation to the history
        self._history.add_entry(current_human_pos, vel, dt)

        trajectories = PredictedTrajectories()
        for i in range(len(sampled_trajectories)):
            path = Path()
            for j in range(len(sampled_trajectories[i])):
                p = sampled_trajectories[i][j]
                pose = PoseStamped()
                pose.header.frame_id = "darias"
                pose.header.stamp = rospy.Time(0)
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = p[2]
                path.poses.append(pose)
            trajectories.trajectories.append(path)
        if self.use_ros:
            self._pub_sampled_trajectories.publish(trajectories)
            self._publish_visualized_trajectories(sampled_trajectories)

            # Publish current direction as a Marker
            self._publish_current_direction_and_goal(self._current_belief)

            ma = MarkerArray()
            for i in range(self._num_goals):
                m = Marker()
                m.header.frame_id = "darias"
                m.header.stamp = rospy.Time.now()
                m.ns = "my_namespace"
                m.id = i + 1000
                m.type = Marker.TEXT_VIEW_FACING
                m.action = Marker.ADD
                m.pose.position.x = self._goals[i][0]
                m.pose.position.y = self._goals[i][1]
                m.pose.position.z = self._goals[i][2] + 0.1
                m.text = "{0:.4f}".format(self._current_belief[i])
                m.pose.orientation.w = 1.0
                m.color.a = 1.0
                m.color.r = 1.0
                m.color.g = 1.0
                m.color.b = 1.0
                m.scale.x = 0.1
                m.scale.y = 0.1
                m.scale.z = 0.1
                ma.markers.append(m)
            self._pub_belief_vis.publish(ma)

        return sampled_trajectories,is_at_goal,avg_vel

    def _generate_trajectories(self, velocity, dt, num_trajectories, prediction_time, current_belief):
        """
        Generates trajectories from the current particles by propagating each of them.
        Parameters
        ----------
        velocity : float
            Current velocity of the human
        dt : float
            Time step
        num_trajectories : int
            Number of trajectories to generate.
        prediction_time : int
            Time steps to predict into the future.
       """




        trajectories = []
        #print("len(self._estimated_positions: {}".format(len(self._estimated_positions)))
        for i in range(num_trajectories):
            # First, sample a position and a goal from the current belief and the estimated positions
            position, goal = self._sample_position(self._estimated_positions, current_belief)

            # Generate a full trajectory
            trajectory = self._generate_trajectory(position, velocity, goal, dt, prediction_time)


            trajectories.append(trajectory)
        return trajectories

    def _sample_position(self, positions, current_belief):
        """
        Samples a positions from the given position vector based on the current belief.

        Parameters
        ----------
        positions : List
            List of positions.

        Returns
        -------
        List :
            np.ndarray : New position based on the belief.<br>
            np.ndarray : Sampled goal
        """
        new_belief = np.copy(current_belief)

        # Threshold Belief and re-normalize
        # If we are very sure of one goal we do not care about the others
        for i in range(self._num_goals):
            if current_belief[i] < self._belief_threshold:
                new_belief[i] = 0.0

        # print "probs belief before:"
        # print new_belief

        # if we are very unsure about one goal we do not use it
        if np.max(new_belief) == 0.0:
            new_belief = np.copy(current_belief)
            for i in range(self._num_goals):
                if current_belief[i] < self._belief_threshold_min:
                    new_belief[i] = 0.0
            print "using old belief above min threshold"

        # this should never happen I think unless we have super many goals
        if np.max(new_belief) == 0.0:
            new_belief = np.copy(current_belief)
            print "using old belief, should not happen"

        # print "probs belief:"
        # print new_belief
        new_belief = new_belief / np.sum(new_belief)


        idx = np.random.choice(a=np.arange(len(positions)), p=new_belief)
        return np.asarray(positions[idx]), self._goals[idx]

    def _generate_trajectory(self, current_pos, velocity, goal, dt, prediction_time):

        """
        Generates a single trajectory up to a certain amount of time steps from the given position.
        Parameters
        ----------
        current_pos :
            Current position of the human.
        velocity :
            Averaged velocity of the human.
        goal :
            Sampled goal of the human.
        dt : float
            Time step
        prediction_time : int
            Amount of time steps to predict into the future.

        Returns
        -------
        np.ndarray : Trajectory
        """

        history = list(np.copy(self._reached_goals))




        out = []
        out.append(np.copy(current_pos))
        first_goal_idx = np.where(self._goals == goal)[0][0]
        selected_goal = goal
        reached_goal = False
        counter_in_goal = 0



        for _ in range(prediction_time):

            # Particle reached selected goal
            # This will continuously chose a next goal, if a particle already reached its predecessor goal

            if np.linalg.norm(current_pos - selected_goal) <= 0.1:
                reached_goal = True






                if counter_in_goal > self.curr_stay_duration_goals[self._is_human_in_range_of_goal(selected_goal)[1]] / dt:

                    selected_goal_idx = np.where(self._goals == selected_goal)[0][0]

                    if len(history) > 0:
                        if not selected_goal_idx == history[-1]:
                            history.append(selected_goal_idx)
                    else:
                        history.append(selected_goal_idx)
                    #print "history:"
                    #print history
                    # Select next goal based on the pre-learned goal-change probabilities


                    #print "selected goal {}".format(selected_goal_idx)
                    probs,available_goals = self.compute_transition_probs_for_goal_from_history(history,selected_goal_idx)


                    for p in probs:
                        if p < self._belief_threshold:
                            p = 0.0

                    print "probs sampling: "
                    print probs / np.sum(np.asarray(probs))
                    selected_goal = self._goals[np.random.choice(available_goals, p=probs / np.sum(np.asarray(probs)))]

                    counter_in_goal = 0.0

                    #print("switching")

                else:
                    counter_in_goal += 1
                    #print("incr counter")





            if reached_goal:
                #print self.curr_stay_duration_goals
                #print self.curr_stay_duration_goals[ self._is_human_in_range_of_goal(selected_goal)[1] ]

                new_pos = self.transition_human(current_pos, velocity, selected_goal, dt)

                out.append(new_pos)
                current_pos = new_pos



            else:
                new_pos = self.transition_human(current_pos, velocity, selected_goal, dt)
                out.append(new_pos)
                current_pos = new_pos

        return np.asarray(out)


    def compute_transition_probs_for_goal_from_history(self, history,selected_goal_idx):
        # compute probabilities for all goals
        available_probs = []
        available_goals = []
        sum_p = 0.0
        for g in range(self._num_goals):
            try:  # TODO remove this since not using dictionary anymore

                # if first_goal_idx == 0:
                #     selected_goal_idx_selection = 0

                if self._goal_change_prob_length > 1:  # will break for more then one goal
                    # print "in if history >1"

                    if not (selected_goal_idx == history[-1]):

                        tmp = np.copy(history[-self._goal_change_prob_length + 1:len(history)])
                    else:
                        if len(history) >= self._goal_change_prob_length + 1:

                            tmp = np.copy(history[
                                          -self._goal_change_prob_length + 1 - 1:len(history) - 1])
                        else:

                            tmp = np.copy(history[0:len(history) - 1])

                    tmp = tmp.tolist()
                    # print"tmp1:"
                    # print tmp
                    #
                    # print "skipping equal belief last history"
                    # else:
                    # print"ok"

                    tmp.append(selected_goal_idx)

                    # print"tmp2:"
                    # print tmp
                    selected_goal_idx_selection = tmp

                    selected_goal_idx_selection.append(g)
                else:
                    # print "in else history <1, selcted goal idx:"
                    # print selected_goal_idx
                    selected_goal_idx_selection = [selected_goal_idx, g]
                # print "reached goals 2:"
                # print self._reached_goals
                # print "sel idx:"
                # print selected_goal_idx_selection
                if len(selected_goal_idx_selection) == self._goal_change_prob_length + 1:
                    prob = self._goal_change_prob[tuple(selected_goal_idx_selection)]





                else:
                    # print "reached:"
                    # print self._reached_goals
                    print "history not yet long enough"
                    idx_history_length = \
                    np.where(self.transition_probabilities_history_vector == len(selected_goal_idx_selection) - 1)[0]

                    if len(idx_history_length) == 1:
                        prob = self._goal_change_prob_all[idx_history_length[0]][tuple(selected_goal_idx_selection)]

                    else:
                        # print "no data known for history size "+ str(len(selected_goal_idx)-1) +" updating with uninformed priors"
                        prob = 1. / (len(self._goals) - 1)

                # print prob
                if prob < 0.0:
                    # print "reached:"
                    # print self._reached_goals
                    # print "_change length:" + str(self._goal_change_prob_length)
                    history_l = np.copy(np.min([self._goal_change_prob_length, len(history)]))
                    history_l -= 1
                    # print "history_l"
                    # print history_l

                    # print "prob"
                    # print prob
                    # print "goal selected idx:"
                    # print selected_goal_idx_selection

                    while prob < 0.0 and history_l >= 1:
                        tmp_idx = tuple(selected_goal_idx_selection[
                                        np.min([self._goal_change_prob_length, len(history)]) - history_l: len(
                                            selected_goal_idx_selection)])

                        prob = self._goal_change_prob_all[history_l - 1][tmp_idx]
                        history_l -= 1

                    if prob < 0.0:
                        print "situation: "
                        print selected_goal_idx_selection
                        print "unknown -> updating with uninformed priors"
                        prob = 1. / (len(self._goals) - 1)
                    else:
                        print "situation: "
                        print selected_goal_idx_selection
                        print "unknown -> using:"
                        print selected_goal_idx_selection[
                              self._goal_change_prob_length - (history_l + 1):len(selected_goal_idx_selection)]
                # else:
                # print "(ok) "
                # print self._reached_goals
                sum_p += prob
                available_goals.append(g)
                available_probs.append(prob)
            except KeyError:
                continue

        probs = [v / sum_p for v in available_probs]

        return probs,available_goals


    def _publish_visualized_trajectories(self, sampled_trajectories):
        marker_array = MarkerArray()
        idx=0
        t=rospy.Time(0)
        for i in range(len(sampled_trajectories)):
            marker = Marker()
            marker.header.frame_id = "darias"
            marker.header.stamp = t
            marker.ns = "my_namespace"
            marker.id = idx
            idx+=1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            for p in sampled_trajectories[i]:
                point = Point()
                point.x = p[0]
                point.y = p[1]
                point.z = p[2]
                marker.points.append(point)

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.005

            marker.color.a = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)


        for i in range(len(sampled_trajectories)):

            for p in sampled_trajectories[i]:
                marker = Marker()
                marker.header.frame_id = "darias"
                marker.header.stamp = t
                marker.ns = "my_namespace"

                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = idx
                idx+=1

                marker.pose.position.x=p[0]
                marker.pose.position.y = p[1]
                marker.pose.position.z = p[2]
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03

                marker.color.a = 0.1
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker_array.markers.append(marker)
        self._pub_trajectory_viz.publish(marker_array)

    def _publish_current_direction_and_goal(self, belief):
        # Get goal with highest belief
        max_g = self._goals[int(np.argmax(belief))]
        # print max_g
        p = PoseStamped()
        p.header.frame_id = "darias"
        p.header.stamp = rospy.Time(0)
        p.pose.position.x = max_g[0]
        p.pose.position.y = max_g[1]
        p.pose.position.z = max_g[2]
        # print type(pose)
        p.pose.orientation.w = 1.0
        self._pub_curr_goal.publish(p)

        human = PoseStamped()
        human.header.frame_id = "darias"
        human.header.stamp = rospy.Time(0)
        human.pose.position.x = self._current_human_pos[0]
        human.pose.position.y = self._current_human_pos[1]
        human.pose.position.z = self._current_human_pos[2]
        human.pose.orientation.w = 1.0
        self._pub_human_pos.publish(human)

        marker = Marker()
        marker.header.frame_id = "darias"
        marker.header.stamp = rospy.Time(0)
        marker.ns = "my_namespace"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        end_p = Point()
        end_p.x = self._current_human_pos[0] + self._direction[0]
        end_p.y = self._current_human_pos[1] + self._direction[1]
        end_p.z = self._current_human_pos[2] + self._direction[2]

        dir = Point()
        dir.x = self._direction[0]
        dir.y = self._direction[1]
        dir.z = self._direction[2]

        # self.pub.publish()

        current_human = Point()
        current_human.x = self._current_human_pos[0]
        current_human.y = self._current_human_pos[1]
        current_human.z = self._current_human_pos[2]
        marker.points.append(current_human)
        marker.points.append(end_p)
        # marker.pose.position.x = res.velocity_human[0]
        # marker.pose.position.y = 1
        # marker.pose.position.z = 1
        # marker.pose.orientation.x = 0.0;
        # marker.pose.orientation.y = 0.0;
        # marker.pose.orientation.z = 0.0;
        # marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self._pub_curr_dir.publish(marker)

    def compute_direction(self, current_observed_human_point):
        """
        Computes the direction the human is currently heading towards.

        Parameters
        ==========
        current_observed_human_point : np.ndarray
            Current position of the human wrist.

        Returns
        =======
        np.ndarray :
            Direction
        """

        if self._history.size() < self._max_belief_history:
            start_idx_history_window = 1
        else:
            start_idx_history_window = self._history.size() - self._max_belief_history

        return current_observed_human_point - self._history.observations[start_idx_history_window - 1]
