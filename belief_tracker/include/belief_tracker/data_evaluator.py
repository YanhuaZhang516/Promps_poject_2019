import numpy as np
from collections import defaultdict
import csv
import matplotlib.pyplot as plt


class DataEvaluator:

    def __init__(self, file_path_data, file_path_goals, true_goals_robot=[],random_seed = 19940704,true_goals=None,idle_dist_h=0.01,idle_dist_r=0.01,idle_threshold=3):
        self.file_path_data = file_path_data
        self.file_path_goals = file_path_goals
        self.goals = np.genfromtxt(file_path_goals, dtype=float, delimiter=",",skip_header=1)[:,0:3]

        if true_goals is not None:

            goals_true = np.genfromtxt(true_goals, dtype=float, delimiter=",", skip_header=1)[:, 0:3]
            new_goals=[]
            for i in range(goals_true.shape[0]):
                idx = np.argmin(np.sum((self.goals-goals_true[i,:])**2,axis=1))

                new_goals.append(np.copy(self.goals[idx]))
            self.goals = np.asarray(new_goals)


        self.true_goals_robot = true_goals_robot
        self.idle_dist_h = idle_dist_h
        self.idle_dist_r = idle_dist_r


        # sort goals to a standard order

        self.trajectory_raw = self.read_trajectory(file_path_data)
        self.trajectory = self.remove_duplicate_positions_dict(self.trajectory_raw)
        self.trajectory = self.cut_start_trajectory(self.trajectory,0.01)
        self.trajectory = self.cut_end_trajectory(self.trajectory,0.01)
        self.trajectory_length = len(self.trajectory["time"])
        self.duration = self.trajectory["time"][-1] - self.trajectory["time"][0]

        self._rest_duration_at_goals = [0 for _ in range(len(self.goals))]  # In time steps
        self._num_close_to_goals = [0 for _ in range(len(self.goals))]



        # ---------------------- Parameters ---------------------------------------
        self.velocity_threshold = 1.0
        self.history_window = 10
        self.threshold_goal_distance = 0.2

        self.velocities = []
        self.windowed_vel = []
        self.at_goal_timed = []
        self.random_seed = random_seed
        self._timesteps_per_second = self.trajectory_length / self.duration

        # -------------------------------------------------------------------------

        #set random seed
        np.random.seed(self.random_seed)

        self.idle_threshold = idle_threshold

    def cut_start_trajectory(self,trajectory, dist):
        out_file = defaultdict(list)
        size = len(trajectory["time"])
        first_move = False
        for t in range(size - 1):
            p1 = trajectory["pos"][t]
            p2 = trajectory["pos"][t + 1]
            if (not first_move) and (np.sqrt(np.sum((p2-p1)**2))>dist):
                first_move = True

            if first_move:
                out_file["time"].append(trajectory["time"][t])
                out_file["pos"].append(trajectory["pos"][t])

        return out_file

    def cut_end_trajectory(self,trajectory,dist):
        out_file = defaultdict(list)
        size = len(trajectory["time"])
        last_move = False
        idx = size-1
        for t in range(size - 2,-1,-1):
            p1 = trajectory["pos"][t]
            p2 = trajectory["pos"][t + 1]
            if  (np.sqrt(np.sum((p2-p1)**2))>dist):
                idx = t
                break

        for t in range(idx):
            out_file["time"].append(trajectory["time"][t])
            out_file["pos"].append(trajectory["pos"][t])

        return out_file


    def read_trajectory(self,file_path):
        """Reads the trajectory from a CSV file.

        Parameters
        ----------
        file_path : str
            Absolute path of the trajectory file.

        Returns
        -------
        defaultdict
            Trajectory as a dictionary, where the keys are ["time", "pos"] and "pos" is a numpy array containing x, y and z
            values.
        """
        trajectory = defaultdict(list)
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader)
            for row in reader:
                trajectory["time"].append(float(row[0]))
                point = []
                for i in xrange(1, len(row)):
                    point.append(float(row[i]))
                trajectory["pos"].append(np.asarray(point))
        return trajectory

    def remove_duplicate_positions_dict(self, trajectory):
        assert isinstance(trajectory, defaultdict)
        out_file = defaultdict(list)
        size = len(trajectory["time"])

        for t in range(size - 1):
            p1 = trajectory["pos"][t]
            p2 = trajectory["pos"][t + 1]
            if not (p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]):
                out_file["time"].append(trajectory["time"][t])
                out_file["pos"].append(trajectory["pos"][t])

        return out_file

    def compute_velocity(self,from_pos, to_pos, dt):
        """
        Computes the velocity of the human wrist moving between two points.

        Parameters
        ----------
        from_pos : np.ndarray
            First position of the human.
        to_pos : np.ndarray
            Next position of the human.
        dt : float
            Delta time between the positions.

        Returns
        -------
        float
            Velocity
        """
        assert isinstance(from_pos, np.ndarray)
        assert isinstance(to_pos, np.ndarray)
        assert isinstance(dt, float)
        return np.linalg.norm(to_pos - from_pos) / dt

    def at_goal(self, position, goal, threshold):
        # type: (np.ndarray, np.ndarray, float) -> bool
        #print np.linalg.norm(goal - position)
        return np.linalg.norm(goal - position) <= threshold

    def compute_transition_probabilities(self, history_lengths):
        at_goal_timed = []

        for t in xrange(1, self.trajectory_length):
            dt = self.trajectory["time"][t] - self.trajectory["time"][t - 1]
            last_pos = self.trajectory["pos"][t - 1]
            current_pos = self.trajectory["pos"][t]

            # Filter velocity
            velocity = self.compute_velocity(last_pos, current_pos, dt)
            if velocity > self.velocity_threshold:
                velocity = self.velocity_threshold - np.random.normal(0, 0.01)
            self.velocities.append(velocity)

            # Compute windowed velocity
            self.windowed_vel.append(np.mean(self.velocities[t - self.history_window:t + 1]))

            # Store, at which goal the human wrist is currently at
            is_at_any_goal = False
            for goal_idx in range(len(self.goals)):
                goal = np.asarray(self.goals[goal_idx])

                if self.at_goal(current_pos, goal, self.threshold_goal_distance):
                    at_goal_timed.append(np.array([t, goal_idx]))
                    is_at_any_goal = True
            if not is_at_any_goal:
                at_goal_timed.append(np.array([t, -1]))

        at_goal_timed = np.asarray(at_goal_timed)
        at_goal_timed_g = at_goal_timed[:, 1]
        # f,ax = plt.subplots(1,1)
        # ax.plot(at_goal_timed_g)
        # plt.show()
        goals_timed_ = at_goal_timed_g[at_goal_timed_g >= 0]
        goals_unique_timed = []

        goals_unique_timed.append(goals_timed_[0])


        for i in range(1, len(goals_timed_)):
            if not goals_timed_[i] == goals_unique_timed[-1]:
                goals_unique_timed.append(goals_timed_[i])

        goals_unique_timed = np.array(goals_unique_timed)

        print goals_unique_timed
        # f, ax = plt.subplots(1, 1)
        # ax.scatter(np.arange(0, len(goals_unique_timed)), goals_unique_timed)
        # plt.show()

        history_transition_data = []

        for i_vec in range(0,len(history_lengths)):
            i=history_lengths[i_vec]

            goal_counts = np.zeros([len(self.goals)] * (i + 1))

            for n in range(i, len(goals_unique_timed)):
                g_past = np.zeros(i)
                idx = []
                for k in range(0, i):
                    g_past[k] = goals_unique_timed[n - (i - k)]
                    idx.append(int(g_past[k]))
                g_current = goals_unique_timed[n]
                idx.append(int(g_current))

                goal_counts[tuple(idx)] += 1

            # sums = np.sum(goal_counts, axis=i)
            # goal_counts / sums

            sums = goal_counts.sum(axis=i, keepdims=1).astype(np.float)
            goal_counts = goal_counts / sums


            goal_counts[np.isnan(goal_counts)] = -1. #0. 1./len(self.goals)

            history_transition_data.append(goal_counts)

        # for i in range(len(self.goals)):
        #     for j in range(len(self.goals)):
        #         print "("+str(i)+","+str(j)+") :"
        #         print history_transition_data[0][i,j]
        #
        # for i in range(len(self.goals)):
        #     for j in range(len(self.goals)):
        #         for k in range(len(self.goals)):
        #             print "(" + str(i) + "," + str(j) + "," + str(k) + ") :"
        #             print history_transition_data[1][i, j, k]

        return history_transition_data

    def compute_rest_durations(self):
        # type: () -> list
        """
        Computes the resting duration of the human at all goals.

        Returns
        -------
        list
            Resting duration for every goal in seconds.
        """
        last_goal_idx = 0

        for t in xrange(1, self.trajectory_length):
            dt = self.trajectory["time"][t] - self.trajectory["time"][t - 1]
            last_pos = self.trajectory["pos"][t - 1]
            current_pos = self.trajectory["pos"][t]

            # Monitor, how long the human rests at the a close goal
            for goal_idx in range(len(self.goals)):
                if self.at_goal(current_pos, self.goals[goal_idx], self.threshold_goal_distance):
                    if last_goal_idx == goal_idx:
                        self._rest_duration_at_goals[goal_idx] += dt
                    else:
                        self._num_close_to_goals[last_goal_idx] += 1
                    last_goal_idx = goal_idx

        # Compute mean over rest duration time steps
        for i in range(len(self._rest_duration_at_goals)):
            if self._num_close_to_goals[i] != 0:
                self._rest_duration_at_goals[i] /= self._num_close_to_goals[i]
            else:
                self._rest_duration_at_goals[i] = 0


        return self._rest_duration_at_goals


    def compute_trajectory_lengths_and_duration_robot(self):
        # type: () -> list
        """
        Computes the resting duration of the human at all goals.

        Returns
        -------
        list
            Resting duration for every goal in seconds.
        """





        dist_total = 0
        dist_one_trial =[]
        dist_one_trial.append(0)




        time_all=0
        time_idle = 0
        time_one_trial=[]
        time_one_trial.append(0)
        finished_pieces = 0
        reached_goal_already = False
        idle_count=0
        self.true_goals_robot = self.trajectory["pos"][0][4:7]

        for t in xrange(1, self.trajectory_length):
            dt = self.trajectory["time"][t] - self.trajectory["time"][t - 1]
            time_all+=dt
            last_pos = self.trajectory["pos"][t - 1][4:7]
            current_pos = self.trajectory["pos"][t][4:7]


            dist_curr = np.linalg.norm(current_pos-last_pos)


            dist_total+=dist_curr
            dist_one_trial[-1]+=dist_curr
            time_one_trial[-1]+=dt

            # Monitor, how long the human rests at the a close goal
            not_at_goal = True
            #for goal_idx in range(len(self.goals)):

            if self.at_goal(current_pos, self.true_goals_robot, self.threshold_goal_distance):
                not_at_goal = False
                if not reached_goal_already:
                    time_one_trial.append(0)
                    dist_one_trial.append(0)


                    finished_pieces += 1
                    reached_goal_already = True


            if not_at_goal:
                reached_goal_already = False

                if dist_curr < self.idle_dist_r and idle_count == self.idle_threshold:
                    time_idle += dt

                elif dist_curr < self.idle_dist_r:
                    idle_count += 1

                else:
                    idle_count = 0




            # if dist_curr < self.idle_dist :#and idle_count == 3:
            #     time_idle += dt
            #     print"idle"


        dist_mean = np.mean(dist_one_trial)
        dist_std = np.std(dist_one_trial)
        time_mean = np.mean(time_one_trial)
        time_std = np.std(time_one_trial)





        return  dist_total, dist_one_trial,dist_mean, dist_std, time_all,time_one_trial,time_mean,time_std,finished_pieces,time_idle



    def compute_trajectory_lengths_and_duration(self):
        # type: () -> list
        """
        Computes the resting duration of the human at all goals.

        Returns
        -------
        list
            Resting duration for every goal in seconds.
        """
        #fig,ax = plt.subplots(2,1)

        idle_over_time = []
        goals_over_time = []
        vel_over_time = []

        last_goal_idx = 0
        dist_total = 0
        dist_one_trial =[]
        dist_one_trial.append(0)

        un_finished_goals = [0,1,2,3]

        time_all=0
        time_one_trial=[]
        time_one_trial.append(0)
        finished_pieces = 0
        time_idle=0
        reached_goal_already = False
        last_reached_goal_idx=-1
        idle_count=0
        for t in xrange(1, self.trajectory_length):
            dt = self.trajectory["time"][t] - self.trajectory["time"][t - 1]
            time_all+=dt
            last_pos = self.trajectory["pos"][t - 1][0:3]
            current_pos = self.trajectory["pos"][t][0:3]

            dist_curr = np.linalg.norm(current_pos-last_pos)
            vel_over_time.append(dist_curr)
            #ax[0].scatter(t,dist_curr)

            dist_total+=dist_curr
            dist_one_trial[-1]+=dist_curr
            time_one_trial[-1]+=dt

            # Monitor, how long the human rests at the a close goal
            not_at_goal = True
            for goal_idx in range(len(self.goals)):

                if self.at_goal(current_pos, self.goals[goal_idx], self.threshold_goal_distance):
                    not_at_goal = False
                    if not reached_goal_already:

                        if goal_idx == 3 and (last_reached_goal_idx != 3):
                            finished_pieces += 1
                        reached_goal_already = True
                        last_reached_goal_idx = goal_idx

                    if goal_idx in un_finished_goals:
                        del un_finished_goals[un_finished_goals.index(goal_idx)]

                    if last_goal_idx == goal_idx :
                        self._rest_duration_at_goals[goal_idx] += dt
                    else:
                        self._num_close_to_goals[last_goal_idx] += 1
                    last_goal_idx = goal_idx

            if not_at_goal:
                reached_goal_already = False

            close_to_goal=False
            for goal_idx in range(len(self.goals)):
                if self.at_goal(current_pos, self.goals[goal_idx], 0.1):
                    close_to_goal =True
                    goals_over_time.append(goal_idx)
            if not close_to_goal:
                goals_over_time.append(-1)
                if dist_curr < self.idle_dist_h and idle_count ==self.idle_threshold :
                    time_idle += dt
                    idle_over_time.append(1)
                    #ax[1].scatter(t, dist_curr)
                elif dist_curr < self.idle_dist_h:
                    idle_count+=1
                    idle_over_time.append(0)
                else:
                    idle_count = 0
                    idle_over_time.append(0)
            else:
                idle_over_time.append(0)

            if len(un_finished_goals)==0:
                dist_one_trial.append(0)
                time_one_trial.append(0)
                #finished_pieces+=1
                un_finished_goals = [0, 1, 2, 3]

        # Compute mean over rest duration time steps
        for i in range(len(self._rest_duration_at_goals)):
            if self._num_close_to_goals[i] != 0:
                self._rest_duration_at_goals[i] /= self._num_close_to_goals[i]
            else:
                self._rest_duration_at_goals[i] = 0

        dist_mean = np.mean(dist_one_trial)
        dist_std = np.std(dist_one_trial)
        time_mean = np.mean(time_one_trial)
        time_std = np.std(time_one_trial)

        # fig, ax = plt.subplots(3, 1)
        # ax[0].plot(idle_over_time)
        # ax[0].set_xlim([0, 2000])
        # ax[1].plot(goals_over_time)
        # ax[1].set_xlim([0, 2000])
        # ax[2].plot(vel_over_time)
        # ax[2].set_ylim([0.0,0.01])
        # ax[2].set_xlim([0, 2000])

        plt.show()
        #plt.show()

        return self._rest_duration_at_goals, dist_total, dist_one_trial,dist_mean, dist_std, time_all,time_one_trial,time_mean,time_std,finished_pieces,time_idle


