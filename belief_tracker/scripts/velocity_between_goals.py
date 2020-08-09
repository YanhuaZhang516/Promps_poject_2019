#!/usr/bin/env python

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import defaultdict

plt.style.use("seaborn")
mpl.rcParams['grid.color'] = "grey"
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.linestyle'] = "dashed"
mpl.rcParams["legend.fancybox"] = True
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.edgecolor"] = "black"


def read_trajectory(file_path):
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


def remove_duplicate_positions_dict(trajectory):
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


def at_goal(position, goal, threshold):
    # type: (np.ndarray, np.ndarray, float) -> bool
    return np.linalg.norm(goal - position) <= threshold


def plot_goals(trajectory, goals):
    plt.figure(100)
    positions = np.asarray(trajectory["pos"])
    x_min = np.min(positions[:, 0])
    x_max = np.max(positions[:, 0])
    y_min = np.min(positions[:, 1])
    y_max = np.max(positions[:, 1])
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.scatter(goals[:, 0], goals[:, 1], marker="o", color="green")
    for g in xrange(len(goals)):
        plt.text(goals[g, 0], goals[g, 1], "Goal {}".format(g))


def get_probability_for_goal_changes(timed_goals):
    flow_diff = np.diff(timed_goals[:, 1])
    goal_traj = []
    for i in range(len(flow_diff)):
        diff = flow_diff[i]
        if diff < 0:
            goal_traj.append(np.array([i + 1, np.abs(diff + 1)]))

    goal_traj = np.asarray(goal_traj)

    time_at_goal = defaultdict(int)
    for i in range(len(goal_traj) - 1):
        if goal_traj[i, 1] != goal_traj[i + 1, 1]:
            time_at_goal["{} to {}".format(goal_traj[i, 1], goal_traj[i + 1, 1])] += 1

    s = float(sum(time_at_goal.values()))
    probs = defaultdict(float)
    for key, value in time_at_goal.iteritems():
        # print("{}: {}".format(key, value / s))
        probs[key] += float(value) / float(s)

    return dict(probs)


def get_timed_vels(time_at_goals):
    timed_vels = defaultdict(list)
    for i in range(len(time_at_goals) - 1):
        if time_at_goals[i][1] != time_at_goals[i + 1][1]:
            timed_vels["{} to {}".format(
                    time_at_goals[i][1],
                    time_at_goals[i + 1][1]
            )].append(time_at_goals[i][0])
    return timed_vels


def timed_goal_trajectory(timed_goals):
    flow_diff = np.diff(timed_goals[:, 1])
    goal_traj = []
    for i in range(len(flow_diff)):
        diff = flow_diff[i]
        if diff < 0:
            goal_traj.append(np.array([i + 1, np.abs(diff + 1)]))

    goal_traj = np.asarray(goal_traj)
    return goal_traj


def get_duration_between_goals(time_at_goals):
    out = defaultdict(list)
    for i in range(len(time_at_goals) - 1):
        if time_at_goals[i][1] != time_at_goals[i + 1][1]:
            out["{} to {}".format(
                    time_at_goals[i][1],
                    time_at_goals[i + 1][1]
            )].append(time_at_goals[i + 1][0] - time_at_goals[i][0])
    return out


def plot_vel_between_goals(timed_vels, duration_between_goals, windowed_vel, from_goal, to_goal, figure_num):
    plt.figure(figure_num)
    plt.xlabel("Time Step")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity from goal {} to goal {}".format(from_goal, to_goal))
    all_velocities = []
    num_vels = len(timed_vels["{} to {}".format(from_goal, to_goal)])
    for i in range(num_vels):
        start_time = timed_vels["{} to {}".format(from_goal, to_goal)][i]
        duration = duration_between_goals["{} to {}".format(from_goal, to_goal)][i] + 10
        velocities = windowed_vel[start_time:start_time + duration + 1]
        all_velocities.append(np.asarray(velocities))
    # Get the minimum length velocity-trace
    min_length = min(map(len, all_velocities))
    all_velocities = np.asarray([v[:min_length] for v in all_velocities])

    mean = np.mean(all_velocities, axis=0)
    std = np.std(all_velocities, axis=0)
    plt.plot(mean, color="C2", label="Mean over {} velocities".format(num_vels))
    plt.fill_between(np.arange(min_length), mean - std, mean + std, alpha=0.5, color="C0", linewidth=1)
    plt.legend()


def main():
    file_path = "/home/albert/trajectories/albert_30_hz_2.csv"
    trajectory = read_trajectory(file_path)
    trajectory = remove_duplicate_positions_dict(trajectory)
    goals = np.genfromtxt("/home/albert/trajectories/goals_user_study_new.csv", dtype=float, delimiter=",",
                          skip_header=1)
    trajectory_length = len(trajectory["time"])
    duration = trajectory["time"][-1] - trajectory["time"][0]
    timesteps_per_second = int(trajectory_length / duration)

    # ---------------------- Parameters ---------------------------------------
    velocity_threshold = 1.0
    history_window = 10
    threshold_goal_distance = 0.10
    velocities = []
    windowed_vel = []
    at_goal_timed = []
    np.random.seed(19940704)
    # -------------------------------------------------------------------------

    for t in xrange(1, trajectory_length):
        dt = trajectory["time"][t] - trajectory["time"][t - 1]
        last_pos = trajectory["pos"][t - 1]
        current_pos = trajectory["pos"][t]

        # Filter velocity
        velocity = compute_velocity(last_pos, current_pos, dt)
        if velocity > velocity_threshold:
            velocity = velocity_threshold - np.random.normal(0, 0.01)
        velocities.append(velocity)

        # Compute windowed velocity
        windowed_vel.append(np.mean(velocities[t - history_window:t + 1]))

        # Store, at which goal the human wrist is currently at
        is_at_any_goal = False
        for goal_idx in range(len(goals)):
            goal = np.asarray(goals[goal_idx])

            if at_goal(current_pos, goal, threshold_goal_distance):
                at_goal_timed.append(np.array([t, goal_idx]))
                is_at_any_goal = True
        if not is_at_any_goal:
            at_goal_timed.append(np.array([t, -1]))

    at_goal_timed = np.asarray(at_goal_timed)
    probability_table = get_probability_for_goal_changes(at_goal_timed)
    time_at_goals = timed_goal_trajectory(at_goal_timed)
    duration_between_goals = get_duration_between_goals(time_at_goals)
    timed_vels = get_timed_vels(time_at_goals)

    plot_vel_between_goals(timed_vels, duration_between_goals, windowed_vel, 0, 1, figure_num=100)
    plot_vel_between_goals(timed_vels, duration_between_goals, windowed_vel, 0, 2, figure_num=200)
    plot_vel_between_goals(timed_vels, duration_between_goals, windowed_vel, 0, 3, figure_num=300)
    plt.show()


if __name__ == '__main__':
    main()
