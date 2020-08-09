#!/usr/bin/python3

import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

plt.style.use('seaborn')


def save_velocities(filepath, vels, avg_vels, mean_vels):
    with open(filepath, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["velocity", "avg_velocity", "mean_velocity"])
        for v, a, m in zip(vels, avg_vels, mean_vels):
            writer.writerow([v, a, m])


def remove_duplicate_positions(trajectory):
    out_file = []
    for t in range(len(trajectory) - 1):
        p1 = trajectory[t][1]
        p2 = trajectory[t + 1][1]
        if not (p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]):
            out_file.append([trajectory[t][0], p1])
    return out_file


def remove_duplicate_positions_dict(trajectory):
    """ Removes duplicate positions, that occur at 2 succeeding time steps.
    Parameters
    ----------
    trajectory : defaultdict
        The whole trajectory with keys ["time (float)", "pos (np.ndarray)"].

    Returns
    -------
    defaultdict
        The cleaned trajectory as a defaultdict.

    """
    assert isinstance(trajectory, defaultdict), "Duplication Removal failed! Trajectory is not a defaultdict!"
    out_file = defaultdict(list)
    size = len(trajectory["time"])

    for t in range(size - 1):
        p1 = trajectory["pos"][t]
        p2 = trajectory["pos"][t + 1]
        if not (p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]):
            out_file["time"].append(trajectory["time"][t])
            out_file["pos"].append(trajectory["pos"][t])

    return out_file


def read_csv(file_name):
    """ Reads a CSV file given its path and returns its content.
    :param file_name (str)  The absolute path to the CSV file.
    :return: The content of the CSV file as an array.
    """
    with open(file_name, "r") as f:
        data = []
        reader = csv.reader(f, delimiter=",")
        next(reader, None)  # Skip header
        for row in reader:
            t = float(row[0])
            pos = np.array([row[1], row[2], row[3]], dtype=float)
            data.append([t, pos])
        return data


def fill_nsec(nsec):
    # type: (str) -> str
    return nsec.rjust(9, '0')


def write_csv(filename, data):
    assert isinstance(filename, str)
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["sec", "nsec", "x", "y", "z"])
        for row in data:
            row[1] = fill_nsec(row[1])
            writer.writerow([row[0], row[1], row[2][0], row[2][2], row[2][2]])


def plot_velocity_interval(velocities, duration, timesteps_per_second, window):
    from_idx = 0
    to_idx = timesteps_per_second * duration

    plt.figure(200)
    plt.title("Assembly time of {}s".format(duration))
    plt.xlabel("Time Steps")
    plt.ylabel("Velocity [m/s]")
    plt.ylim([-0.1, np.max(velocities[0]) + 0.5])
    plt.plot(velocities[0][from_idx:to_idx], label="Instantaneous Velocity")
    plt.plot(velocities[1][from_idx:to_idx], color="C2", label="Windowed Velocity: {} Time Steps".format(window))
    plt.plot(velocities[2][from_idx:to_idx], color="C1", label="Mean")

    plt.legend()
    plt.grid(linestyle='dotted', color="gray", alpha=10)
    leg = plt.legend(frameon=True, loc=2)
    leg.get_frame().set_linewidth(1.0)


def plot_beliefs(beliefs, timesteps_per_second, duration, time_steps):
    to_idx = timesteps_per_second * duration

    plt.figure(300)
    plt.title("Belief over Goals for {} seconds".format(duration))
    plt.xlabel("Time Steps")
    plt.ylabel("Probability")
    for goal in range(len(beliefs[0])):
        # plt.plot(np.arange(time_steps)[:to_idx], beliefs[:to_idx, goal], label="Goal {}".format(goal + 1))
        plt.plot(beliefs[:to_idx, goal], label="Goal {}".format(goal + 1))

    plt.legend()
    plt.grid(linestyle='dotted', color="gray", alpha=10)
    leg = plt.legend(frameon=True, loc=2)
    leg.get_frame().set_linewidth(1.0)
