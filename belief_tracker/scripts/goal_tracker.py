#! /usr/bin/env python
import time

import tf
import rospy
import numpy as np
import goal_tracker_visualization as gtv

from promp_robot_control_msgs.msg import Start
from geometry_msgs.msg import Point
from promp_robot_control_msgs.srv import Goals, GoalsRequest, GoalsResponse

from numpy import genfromtxt
import rospkg

import pickle


class GoalTracker:
    def __init__(self):
        self.start_belief_sub = rospy.Subscriber("/start_goal_tracker", Start, self.StartCallback)
        self.goals_service = rospy.Service("/gt_get_goals", Goals, self.GetGoalsCallback)

        self.compute_goals_from_csv_service = rospy.Service("/goal_tracker/get_goals_from_csv", Goals, self.ComputeGoalsFromCsvCallback)
        self.reset_goals_from_csv_service = rospy.Service("/goal_tracker/reset_goals_from_csv", Goals,
                                                            self.ResetGoalsFromCsvCallback)

        self.tf_listener = tf.TransformListener()

        # Visualization Class
        self.gtv = gtv.GoalTrackerVisualization()

        self.vis_timer = rospy.Timer(rospy.Duration(1.0), self.vis_tick)





        self.resetGoalTracker()

        self.min_support_for_goal = 2

        self.min_diff_positions_after_each_other = 0.02
        self.last_position=[]



        # read goals from yaml file
        self.pkg_path = rospkg.RosPack().get_path('belief_tracker')
        goal_file=rospy.get_param('/goal_tracker_node/goal_file')
        print goal_file

        if goal_file == "":
            self.update_goals = True
            print "no goals"

        else:
            print goal_file
            self.update_goals = False
            goal_file=str(self.pkg_path) + "/data/" + goal_file
            self.reset_goals_from_csv(goal_file)



            #print self.goal_means.shape


    def ResetGoalsFromCsvCallback(self, request):
        self.resetGoalTracker()
        print "resetting goal tracker and stopping online mode"
        self.update_goals = False
        print request.csv_filename
        self.reset_goals_from_csv(request.csv_filename)

        response = GoalsResponse()
        # Convert numpy goal to ROS point goal and append to response

        for goal in self.cached_goals:
            g = Point()
            g.x = goal[0]
            g.y = goal[1]
            g.z = goal[2]
            response.goals.append(g)
        return response



    def reset_goals_from_csv(self, goal_file):
        data_array = np.genfromtxt(goal_file, dtype=float, delimiter=",",skip_header=1)

        # self.goal_means = data_array[1:data_array.shape[0]]
        print data_array.shape
        for i in range(0, data_array.shape[0]):
            self.goal_means.append(data_array[i,0:3])
            self.goal_covs.append(np.diag(data_array[i,3:6]))
            #self.goal_covs.append(np.eye(3) * self.init_sigma_file)
        print self.goal_means

        self.n_comp = len(self.goal_means)
        self.cached_goals = self.goal_means

        # self.Tick()
    def resetGoalTracker(self):
        # Online Goal Updater essentials
        self.all_goals=[]
        self.v=[]
        self.last_position = []
        self.goals_changed = False
        self.last_observed_pose = []
        self.last_time = []
        self.last_direction = []
        self.last_direction_obs = []

        self.obs_window = 5
        self.current_points = []
        self.current_times = []

        self.min_vel_goal = 0.1#0.05
        self.goal_means = []
        self.cached_goals = []
        self.goal_covs = []
        self.n_comp = 0
        self.init_sigma = 0.02#0.05
        self.sp = []
        self.T_nov = 1.0
        self.min_cov = 0.001
        self.new_stop_flag = True


        self.init_sigma_file = 0.01

        self.block_vis=False

    # def __del__(self):
    #     self.goals_service.shutdown("GoalTracker closed!")

    def vis_tick(self, msg):
        if not self.block_vis:
            self.gtv.update_goal_visualization(self.n_comp, self.goal_means, self.goal_covs)


    def StartCallback(self, msg):
        # type: (Start) -> None
        """This function starts the GoalTracker, when the respective message is published."""
        if msg.start:
            timer = rospy.Timer(rospy.Duration(0.03), self.Tick)


    def GetGoalsCallback(self, request):
        # type: (GoalsRequest) -> GoalsResponse
        """
        Caller will get the current goals from the Goal Tracker. Flag for changed goals will be set to false.
        :param request: Empty
        :return: Current goals
        """
        response = GoalsResponse()
        # Convert numpy goal to ROS point goal and append to response


        for goal in self.cached_goals:
            g = Point()
            g.x = goal[0]
            g.y = goal[1]
            g.z = goal[2]
            response.goals.append(g)
        return response

    def ComputeGoalsFromCsvCallback(self, request):
        self.resetGoalTracker()
        print "Stoping online updating of goals due to csv file input"
        self.update_goals = False

        my_data = genfromtxt(request.csv_filename, delimiter=',')[1:-1,:] # cutting off the header [t,x,y,z]
        times=my_data[:,0]
        human_positions=my_data[:,1:4]

        for i in range(times.shape[0]):
            self.update_step(human_positions[i],times[i] )
            #time.sleep(0.01)

        response = GoalsResponse()


        # Convert numpy goal to ROS point goal and append to response
        self.block_vis = True
        idces_delete=[]
        for i in reversed(xrange(len(self.cached_goals))):
            print i

            print self.sp[i]

            if self.sp[i]<self.min_support_for_goal:
                idces_delete.append(i)
                del self.sp[i]

                del self.goal_means[i]
                del self.goal_covs[i]
                del self.v[i]


                #del self.cached_goals[i]
                self.n_comp -= 1


                print "deleting goal with too mall support"
        self.gtv.update_goal_visualization(self.n_comp, self.goal_means, self.goal_covs)
        self.block_vis = False




        for i in range(len(self.cached_goals)):
            g = Point()

            g.x = self.cached_goals[i][0]
            g.y = self.cached_goals[i][1]
            g.z = self.cached_goals[i][2]
            response.goals.append(g)


            g = Point()
            g.x = np.diag(self.goal_covs[i])[0]
            g.y = np.diag(self.goal_covs[i])[1]
            g.z = np.diag(self.goal_covs[i])[2]
            response.covariances.append(g)

        goal_data = {
            'all_goals': self.all_goals
        }
        filename = self.pkg_path + "/tmp_all_goals.pkl"
        output = open(filename, 'wb')
        pickle.dump(goal_data, output)
        return response











    def UpdateGoals(self, data):
        # type: (np.ndarray) -> None
        if self.n_comp == 0:
            self.goal_means = [data]
            self.goal_covs.append(np.eye(3) * self.init_sigma)
            self.n_comp += 1
            self.sp = [1.0]
            self.v = [1.0]
        else:
            p_x_given_j = np.zeros(self.n_comp)
            p_x_given_j_2 = np.zeros(self.n_comp)
            thresh = np.zeros(self.n_comp)
            for j in range(self.n_comp):
                # Get current component parameters
                Cj = self.goal_covs[j]
                uj = self.goal_means[j]

                # Compute probability
                xu = (data - uj).reshape(-1, 1)
                b = -0.5 * np.dot(xu.T, np.dot(np.linalg.pinv(Cj), xu))
                n_dim = 3.0
                a = (2 * np.pi) ** (n_dim / 2.0) * np.sqrt(np.linalg.det(Cj))
                p_x_given_j[j] = 1 / a * np.exp(b)

            if np.all(np.less(p_x_given_j, self.T_nov)):
                # print('new comp !!!')
                self.n_comp += 1
                self.goal_means.append(data)  # component mean
                self.goal_covs.append(np.eye(3) * self.init_sigma)

                self.v.append(1.0)  # age of component j
                self.sp.append(1.0)  # accumulator of component j
                # self.weights_.append(1.0 / self.n_comp)

            else:
                sum_pj = sum(p_x_given_j)
                p_j_given_x = [p_x_given_j[j] / sum_pj for j in range(self.n_comp)]

                for j in range(self.n_comp):
                    self.v[j] += 1.0
                    self.sp[j] += p_j_given_x[j]

                    wj = p_j_given_x[j] / self.sp[j]
                    mean_old = np.copy(self.goal_means[j])
                    mean_old_e = np.copy(self.goal_covs[j])
                    self.goal_means[j] = self.updateMean(self.goal_means[j], data, wj)
                    self.goal_covs[j] = self.updateCov(self.goal_covs[j], data, mean_old, self.goal_means[j],
                                                       wj, self.sp[j], p_j_given_x[j])
                    for c in range(0, 3):
                        # print self.goal_covs[j][c][c]
                        if self.goal_covs[j][c][c] < self.min_cov:
                            self.goal_covs[j][c][c] = self.min_cov

        self.cached_goals = self.goal_means


    def updateMean(self, m_old, x, w):
        ej = x - m_old
        d_uj = w * ej
        return m_old + d_uj


    def updateCov(self, C_old, x, m_old, m, w, sp, pjx):
        #ej = x - m_old
        #d_uj = w * ej
        ej_star = x - m

        #wj_s = w + np.exp(-sp) * pjx

        help = m-m_old

        #wj_s = w #+ np.exp(-sp) * pjx
        # wj_s = w + 0.5 * (1 / sp) * pjx
        # This is causing problems maybe

        #cov_new = (1 - wj_s) * C_old + wj_s * np.outer(ej_star, ej_star) - (wj_s - w) * np.outer(d_uj, d_uj)

        cov_new = (1 - w) * C_old + w * np.outer(ej_star, ej_star) - np.outer(help, help)

        try:
            cov_new = self.nearestSPD(cov_new)
        except:
            print'++++++++++++++++++ ERROR in svd !!!!!!!!!!!!!!!!!!!!'
            cov_new = cov_new + np.eye(cov_new.shape[0]) * 1e-6
            cov_new = self.nearestSPD(cov_new)

        return cov_new


    def Tick(self, msg):
        if self.update_goals:
            """
            This function will be executed as long as the node runs. It will lookup the human positions and calls the update functions for the goals.
            """

            human_position = []
            current_time = 0.0
            try:
                (human_position, rotation) = self.tf_listener.lookupTransform("/darias", "/human", rospy.Time(0))
                current_time = time.time()
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("No Human Position received")
                return

            self.update_step( human_position, current_time)


    def update_step(self,human_position, current_time):
            if len(self.current_points) < self.obs_window:
                self.current_points.append(human_position)
                self.current_times.append(current_time)
            else:
                del self.current_points[0]
                del self.current_times[0]

                self.current_points.append(human_position)
                self.current_times.append(current_time)

                vel_human = self.CalculateHumanVelocity()
                if vel_human > self.min_vel_goal:
                    if not self.new_stop_flag:
                        self.new_stop_flag = True
                else:
                    if self.new_stop_flag:
                        if len(self.last_position)>0:
                            diff = np.sum((self.last_position-human_position)**2)
                        else:
                            diff = 100

                        if diff > self.min_diff_positions_after_each_other:
                            print diff
                            print self.v
                            print self.sp
                            self.gtv.add_vel_goal_candidate(human_position)
                            self.UpdateGoals(np.asarray(human_position))
                            self.last_position = np.copy(human_position)
                            self.gtv.update_goal_visualization(self.n_comp, self.goal_means, self.goal_covs)

                            self.all_goals.append([np.copy(self.goal_means),np.copy(self.goal_covs),np.copy(human_position)])

                            # rospy.loginfo("Num Goals: {}".format(self.n_comp))
                        self.new_stop_flag = False


    def CalculateHumanVelocity(self):
        # type: () -> float
        """Computes the velocity of the human over the last observations.
        The number of last observations is specified in 'self.obs_window'

        :return: Velocity in m/s
        """
        dist = 0.0
        for i in range(1, self.obs_window):
            p1 = self.current_points[i - 1]
            p2 = self.current_points[i]
            dist += np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

        dt = self.current_times[self.obs_window - 1] - self.current_times[0]
        return dist / dt


    def nearestSPD(self, A):
        """
        % nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
        % usage: Ahat = nearestSPD(A)
        %
        % From Higham: "The nearest symmetric positive semidefinite matrix in the
        % Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
        % where H is the symmetric polar factor of B=(A + A')/2."
        %
        % http://www.sciencedirect.com/science/article/pii/0024379588902236
        %
        % arguments: (input)
        %  A - square matrix, which will be converted to the nearest Symmetric
        %    Positive Definite Matrix.
        %
        % Arguments: (output)
        %  Ahat - The matrix chosen as the nearest SPD matrix to A.
        """
        A = np.atleast_2d(A)

        # test for a square matrix A
        r, c = A.shape
        if r != c:
            print("A has to be symmetric")
            return A
        elif (r == 1) and (A <= 0):
            # A was scalar and non-positive, so just return eps
            Ahat = np.spacing(1)
            return Ahat

        # symmetrize A into B
        B = (A + A.T) / 2

        # Compute the symmetric polar factor of B. Call it H.
        # Clearly H is itself SPD.
        U, Sigma, Vh = np.linalg.svd(B)
        Sigma = np.diag(Sigma)
        V = Vh.T

        H = np.dot(V, np.dot(Sigma, V.T))

        # get Ahat in the above formula
        Ahat = (B + H) / 2

        # ensure symmetry
        Ahat = (Ahat + Ahat.T) / 2

        # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
        is_positive = np.all(np.linalg.eigvals(Ahat) > 0)
        k = 0
        count = 0
        while not is_positive and count < 1000:
            count += 1
            is_positive = np.all(np.linalg.eigvals(Ahat) > 0)
            k = k + 1
            if not is_positive:
                # Ahat failed the chol test. It must have been just a hair off,
                # due to floating point trash, so it is simplest now just to
                # tweak by adding a tiny multiple of an identity matrix.
                mineig = min(np.real(np.linalg.eigvals(Ahat)))
                # print(mineig)
                # print(A)
                Ahat = Ahat + (-mineig * k ** 2 + np.spacing(mineig)) * np.identity(A.shape[0])

        # disp([ 'nearestSPD iterations: ' num2str(count) ]);
        return Ahat


if __name__ == '__main__':
    rospy.init_node("goal_tracker")
    gt = GoalTracker()
    rospy.spin()
