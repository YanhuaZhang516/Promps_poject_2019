#!/usr/bin/env python

import tf
import csv
import rospy
import rospkg

class CsvToTf:
    def __init__(self):
        self.pkg_path = rospkg.RosPack().get_path('belief_tracker')
        filename = self.pkg_path+"/recorded_data/userstudy_all/H14L/H14L_no_robot.csv"
        # TODO: END

        self.trajectory = self.get_trajectory_from_csv(filename)
        self.trajectory.reverse()
        self.frequency = self.get_frequency_from_csv(filename)

        self.rate = rospy.Rate(self.frequency)

        print 1./self.frequency
        self.timer = rospy.Timer(rospy.Duration(1.0* 1./self.frequency), self.tik)

    def tik(self,event):
        print "tick"
        if len(self.trajectory) > 0:
            self.broadcast_tf(self.trajectory.pop())

    #     broadcast_tf(trajectory.pop())


    def get_frequency_from_csv(self,csv_filepath):
        # type: (str) -> float
        with open(csv_filepath, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            next(csv_reader)

            total_time = 0.0
            entries = 0
            for row in csv_reader:
                entries += 1
                total_time = float(row[0])

            return entries / total_time


    def get_trajectory_from_csv(self,csv_filepath):
        # type: (str) -> list
        with open(csv_filepath, "r") as csv_file:
            trajectory = []
            csv_reader = csv.DictReader(csv_file, delimiter=",")
            for row in csv_reader:
                trajectory.append([float(row["human_x"]), float(row["human_y"]), float(row["human_z"])])

            return trajectory


    def broadcast_tf(self,point):
        print "tf broadcast"
        br = tf.TransformBroadcaster()
        br.sendTransform((point[0], point[1], point[2]),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "/human",
                         "/darias")


def main():
    rospy.init_node("csv_to_tf_node")
    # TODO: EDIT ME

    bla = CsvToTf()
    rospy.spin()
    #filename = "/home/albert/adda_ros/src/pomdp_collision/belief_tracker/recorded_data/test_data_RL.csv"
    #filename = "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/recorded_data/test_data_RL.csv"
    # TODO: END
    #
    # trajectory = get_trajectory_from_csv(filename)
    # trajectory.reverse()
    # frequency = get_frequency_from_csv(filename)
    #
    # rate = rospy.Rate(frequency)
    # while not rospy.is_shutdown() and len(trajectory) > 0:
    #     broadcast_tf(trajectory.pop())
    #     rate.sleep()


if __name__ == '__main__':
    main()
