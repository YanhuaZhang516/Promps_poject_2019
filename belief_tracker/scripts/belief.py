#!/usr/bin/python

import csv
import belief_io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from include.belief_tracker.history import *
from collections import defaultdict
from scipy.stats import multivariate_normal

plt.style.use("seaborn")
mpl.rcParams['grid.color'] = "grey"
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.linestyle'] = "dashed"


def normalize(v):
    """Normalizes a numpy vector.

    Parameters
    ----------
    v : np.ndarray
        Vector to normalize.

    Returns
    -------
    np.ndarray
        Normalized NumPy vector.
    """
    assert isinstance(v, np.ndarray)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm




class Belief:
    def __init__(self, goals, init_method="uniform"):
        """
        Initializes the belief.

        Parameters
        ----------
        goals : np.ndarray
            Vector of goals.

        init_method : str, optional
            Desired initialization method:
            `uniform`: Sets the probabilities for each possible variable
            uniformly.
            `random`: Sets random probabilities for each variable.
        """
        assert isinstance(goals, np.ndarray)
        assert len(goals) > 0, "[Belief] Vector of goals must be > 0!"

        self._goal_change_prob = {'1 to 0': 0.2328767123287671, '2 to 0': 0.1917808219178082,
                                  '3 to 0': 0.0273972602739726, '3 to 1': 0.1095890410958904,
                                  '0 to 1': 0.1095890410958904, '0 to 3': 0.136986301369863,
                                  '0 to 2': 0.1917808219178082}

        self._method = init_method
        self._goals = goals
        self._num_goals = len(goals)
        self._current_belief = []
        self._current_est_positions = []
        self._observation_variance = 0.0001
        self._human_standard_deviation = 0.005
        self._history_belief = []
        self._max_history = 30
        self._history = History()

        self._current_belief = self._init_belief()
        self._history_belief.append(self._current_belief)

    def _init_belief(self):
        """
        Initializes the Belief based on the type of initialization method.
        """
        if self._method == "uniform":
            return np.ones((self._num_goals,)) * 1.0 / self._num_goals
        else:
            random_nums = np.random.uniform(size=self._num_goals)
            random_nums /= np.sum(random_nums)
            belief = random_nums
            return belief

    def _transition_human(self, position, velocity, goal, dt):
        """
        Computes the new position of the human wrist based on the current position,
        velocity and the headed goal.

        Parameters
        ----------
        position : np.ndarray
            Current position of the human wrist.
        velocity : float
            Instantaneous velocity.
        goal : np.ndarray
            Headed goal of the human wrist.
        dt : float
            Time step

        Returns
        -------
        np.ndarray
            New position of the human wrist.
        """
        noise = np.random.normal(loc=0.0, scale=self._human_standard_deviation, size=3)
        change_in_position = (velocity * dt * normalize(goal - position)) + noise
        return position + change_in_position

    def _observation_likelihood(self, current_pos, last_pos, goal, vel, dt):
        """
        Computes the observation likelihood for the current observation by incorporating the
        last observation calculating the next position given the human transition model.

        Parameters
        ----------
        current_pos : np.ndarray
            Current observation of the human position.
        last_pos : np.ndarray
            Last observation of the human position.
        goal : np.ndarray
            Goal of the human.
        vel : float
            Velocity
        dt : float
            Delta time

        Returns
        -------
        float
            Likelihood of the current observation.

        """
        estimated_position = self._transition_human(last_pos, vel, goal, dt)
        return multivariate_normal.pdf(x=current_pos,
                                       mean=estimated_position,
                                       cov=np.identity(3) * self._observation_variance), estimated_position

    @staticmethod
    def compute_velocity(from_pos, to_pos, dt):
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

    def update_belief(self, last_pos, observation, velocity, dt):
        # First observation
        if self._history.size() < 1:
            self._current_belief, self._current_est_positions = self._update_belief_once(last_pos, observation,
                                                                                         velocity, dt,
                                                                                         self._current_belief)
            self._history.add_entry(observation, velocity, dt)
            return [self._current_belief, self._current_est_positions]

        # Only one observation in history
        if self._history.size() == 1:
            self._current_belief, self._current_est_positions = self._update_belief_once(last_pos, observation,
                                                                                         velocity, dt,
                                                                                         self._current_belief)
            self._history.add_entry(observation, velocity, dt)
            return [self._current_belief, self._current_est_positions]

        # Update belief over history first
        new_belief = self._update_belief_over_history()

        # Update with new observation
        last_obs = np.asarray(self._history.last_entry()[0])
        new_belief, self._current_est_positions = self._update_belief_once(last_obs, observation, velocity, dt,
                                                                           new_belief)
        self._current_belief = new_belief
        self._history.add_entry(observation, velocity, dt)
        self._history_belief.append(self._current_belief)

        return [self._current_belief, self._current_est_positions]

    def _update_belief_once(self, last_pos, observation, velocity, dt, old_belief):
        """
        Updates the belief based on the current belief and observation.
        
        Parameters
        ----------
        last_pos : np.ndarray
            Last position of the human.
        observation : np.ndarray
            Current observation of the human.
        velocity : float
            Current velocity of the human
        dt : float
            Delta time for the transition.

        Returns
        -------
        list :
            list : The updated belief <br>
            list : Updated particles <br>
            list : New estimated positions of the human wrist.
        """
        new_belief = np.empty_like(old_belief)
        normalization_factor = 0.0
        likelihoods = []
        est_positions = []

        # Compute normalization factor and store likelihoods and estimated positions
        for goal in range(self._num_goals):
            likelihood, est_position = self._observation_likelihood(observation, last_pos, self._goals[goal], velocity,
                                                                    dt)
            normalization_factor += likelihood * old_belief[goal]
            likelihoods.append(likelihood)
            est_positions.append(est_position)

        # Compute new belief for all goals
        for goal in range(self._num_goals):
            probability = (likelihoods[goal] * old_belief[goal]) / normalization_factor
            new_belief[goal] = probability

        # self._current_belief = new_belief
        self._current_est_positions = est_positions

        return [new_belief, est_positions]

    def _update_belief_over_history(self):
        # type: () -> list
        """
        Updates the belief from the last 'self._max_history' time steps.
        History has to have at least 2 entries!
        The current observation will not be included here!
        :return: The belief over the history of length 'self._max_history'
        """
        belief = self._init_belief()

        if self._history.size() < self._max_history:
            start_idx_history_window = 1
        else:
            start_idx_history_window = self._history.size() - self._max_history

        for t in range(start_idx_history_window, self._history.size()):
            dt = self._history.dts[t]
            current_pos = np.asarray(self._history.observations[t])
            last_pos = np.asarray(self._history.observations[t - 1])
            vel = self.compute_velocity(last_pos, current_pos, dt)
            avg_vel = self._history.last_entry()[1]
            belief, _ = self._update_belief_once(last_pos, current_pos, avg_vel, dt, belief)

        return belief

    def _sample_position(self, positions):
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
        idx = np.random.choice(a=np.arange(len(positions)), p=self._current_belief)
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
            Goal of the human.
        dt : float
            Time step
        prediction_time : int
            Amount of time steps to predict into the future.

        Returns
        -------
        np.ndarray : Trajectory
        """
        out = []
        selected_goal = goal
        changed_goal = False
        for _ in range(prediction_time):
            if np.linalg.norm(current_pos - selected_goal) <= 0.1:
                changed_goal = True
                available_goals = []
                available_probs = []
                sum_p = 0.0
                for g in range(self._num_goals):
                    try:
                        selected_goal_idx = np.where(self._goals == selected_goal)[0][0]
                        prob = self._goal_change_prob["{} to {}".format(selected_goal_idx, g)]
                        sum_p += prob
                        available_goals.append(g)
                        available_probs.append(prob)
                    except KeyError:
                        continue
                probs = [v/sum_p for v in available_probs]
                selected_goal = self._goals[np.random.choice(available_goals, p=probs)]
            if changed_goal:
                # TODO: I changed heeereee
                new_pos = self._transition_human(current_pos, velocity, selected_goal, dt)
            else:
                new_pos = self._transition_human(current_pos, velocity, selected_goal, dt)
            out.append(new_pos)
            current_pos = new_pos

        return np.asarray(out)

    def generate_trajectories(self, velocity, dt, num_trajectories, prediction_time):
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
        for i in range(num_trajectories):
            position, goal = self._sample_position(self._current_est_positions)
            trajectory = self._generate_trajectory(position, velocity, goal, dt, prediction_time)
            trajectories.append(trajectory)
        return trajectories


def read_trajectory(filepath):
    """
    Reads the trajectory from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Absolute path of the trajectory file.
    
    Returns
    -------
    defaultdict
        Trajectory
    """
    t = defaultdict(list)
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        for row in reader:
            t["time"].append(float(row[0]))
            point = []
            for i in range(1, len(row)):
                point.append(float(row[i]))
            t["pos"].append(np.asarray(point))
    return t


def print_statistics_dict(trajectory, name):
    """Prints information about the trajectory.
    
    Parameters
    ----------
    trajectory : defaultdict
        The duplicate-free trajectory.
    name : str
        Name of the person who performed the trajectory.
    
    """

    trajectory_length = len(trajectory["time"])
    trajectory_duration = trajectory["time"][-1] - trajectory["time"][0]
    timesteps_per_second = int(trajectory_length / trajectory_duration)

    print("----------------- Pre-Statistics for {} ----------------------------------".format(name))
    print("Size without duplicates:     {} points".format(trajectory_length))
    print("Duration:                    {} seconds".format(trajectory_duration))
    print("Frequency:                   {} timesteps / second".format(timesteps_per_second))
    print("--------------------------------------------------------------------------".format(name))


def get_trajectory_from_name(user_name):
    """
    Loads the trajectory given the name of the user and pre-processes it by removing duplicate positions.

    Parameters
    ----------
    user_name : str
        The users' trajectory.

    Returns
    -------
    np.ndarray
        The pre-processed trajectory.

    """
    trajectory_filepath = "/home/albert/trajectories/{}.csv".format(user_name)
    trajectory = read_trajectory(trajectory_filepath)

    # Remove duplicates
    trajectory = belief_io.remove_duplicate_positions_dict(trajectory)
    print_statistics_dict(trajectory, user_name)

    return trajectory


def plot_goals(trajectory, goals, belief, seconds_passed):
    plt.figure(100)
    plt.xlim([0, 2])
    plt.ylim([-0.6, 0.6])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.scatter(goals[:, 0], goals[:, 1], marker="o", color="C2")
    for g in range(len(goals)):
        x = goals[g, 0]
        y = goals[g, 1]
        plt.text(x - 0.15, y + 0.1, "Goal {}".format(g), alpha=0.8)
        plt.text(x - 0.10, y + 0.05, "{0:.4f}".format(belief[g]), alpha=0.8)
    plt.text(1.6, 0.5, "{0:.4f} seconds".format(seconds_passed), alpha=0.8)


def clear_plot(trajectory, goals, belief):
    plt.clf()
    # plot_goals(trajectory, goals, belief)


def main():
    """
    Entry Point of the module.
    """
    # name = "Yi_1_30_Hz"
    name = "albert_30_hz_2"
    goals_filepath = "/home/albert/trajectories/goals_user_study_new.csv"
    goals = np.genfromtxt(goals_filepath, dtype=float, delimiter=',', skip_header=1)
    belief = Belief(goals, init_method="uniform")
    trajectory = get_trajectory_from_name(name)

    # -------------------------- Parameter -------------------------------------
    time_steps = len(trajectory["time"])
    duration = trajectory["time"][-1] - trajectory["time"][0]
    timesteps_per_second = int(time_steps / duration)
    velocity_threshold = 1.0
    vel_history_window = 10
    plot_duration = 30  # For plotting velocity; In seconds
    animate = True
    animation_start_time = 1
    animation_pause_time = 0.00001
    prediction_interval = 2.0  # in seconds
    num_sampled_trajectories = 10
    # -------------------------- Parameter -------------------------------------

    beliefs = []
    velocities = []
    windowed_vel = []

    t_predict = int(prediction_interval * timesteps_per_second)

    plot_goals(trajectory, goals, [0, 0, 0, 0], 0)

    time_steps = 400
    for t in range(1, time_steps):
        last_pos = trajectory["pos"][t - 1]
        current_pos = trajectory["pos"][t]
        dt = trajectory["time"][t] - trajectory["time"][t - 1]

        if t % 50 == 0:
            print("{} time steps done!".format(t))

        velocity = belief.compute_velocity(last_pos, current_pos, dt)

        # Clip velocity if it's too high, and add some noise to it
        if velocity >= velocity_threshold:
            velocity = velocity_threshold - np.random.normal(0, 0.01)
        velocities.append(velocity)

        # Compute windowed velocity
        avg_vel = float(np.mean(velocities[t - vel_history_window:t + 1]))
        windowed_vel.append(avg_vel)

        # Update belief and get particles, as well as possible positions from the new
        [updated_belief, _] = belief.update_belief(last_pos, current_pos, 0.6, dt)

        sampled_trajectories = belief.generate_trajectories(0.6, dt, num_sampled_trajectories, t_predict)

        beliefs.append(updated_belief)

        # Plot as animation
        if t > animation_start_time and animate:
            clear_plot(trajectory, goals, updated_belief)
            for i in range(len(sampled_trajectories)):
                plt.plot(sampled_trajectories[i][:, 0], sampled_trajectories[i][:, 1], color="black",
                         alpha=0.2)
            plt.scatter(current_pos[0], current_pos[1], marker="o", c="C0")
            plot_goals(trajectory, goals, updated_belief, t / float(timesteps_per_second))
            plt.pause(animation_pause_time)

    mean_velocities = np.ones_like(velocities) * np.mean(velocities)
    beliefs = np.asarray(beliefs)
    belief_io.plot_velocity_interval([velocities, windowed_vel, mean_velocities], plot_duration, timesteps_per_second,
                                     vel_history_window)
    belief_io.plot_beliefs(beliefs, timesteps_per_second, plot_duration, time_steps)

    plt.show()


if __name__ == '__main__':
    SEED = 19940407
    np.random.seed(SEED)
    main()
