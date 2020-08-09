#! /usr/bin/env python
import time

import tf
import rospy
import numpy as np
from numpy import genfromtxt
import rospkg
import csv
import time
from std_msgs.msg import Empty, String,Bool
import matplotlib

import matplotlib.pyplot as plt

from numpy import genfromtxt

import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class HumanTrajectoryRecording:
    def __init__(self):
        self.recording = False
        self.rec_restarted = False



        self.recorded_data_human=[]
        self.recorded_data_endeff = []
        self.recorded_data_joints = []

        self.record_robot = False

        self.current_joints = []

        self.recorded_times = []

        self.tf_listener = tf.TransformListener()
        self.rec_timer = rospy.Timer(rospy.Duration(0.03), self.recording_loop)
        #self.p_timer = rospy.Timer(rospy.Duration(0.03), self.recording_loop)

        self.filename = "default_csv_file"
        rospy.Subscriber("/human_trajectory_recording/start_recording", Bool, self.start_recording_callback) # bool indicates if we record robot also
        rospy.Subscriber("/human_trajectory_recording/stop_recording", String, self.stop_recording_callback)
        rospy.Subscriber("/joint_states", JointState, self.joints_callback)

        self.visualization_publisher = rospy.Publisher('/human_trajectory_recording/visualize_recording', String,queue_size=1)
        self.timer_status = rospy.Timer(rospy.Duration(5.), self.auto_save_loop)


    def joints_callback(self,msg):
        self.current_joints = msg.position

    def start_recording_callback(self, msg):
        #plt.close(self.f)
        self.record_robot = msg.data
        self.rec_restarted = True

        if self.record_robot:
            print "start recording human and robot...."
        else:
            print "start recording human ..."

        self.recorded_data_human = []
        self.recorded_times = []
        self.recorded_data_endeff = []
        self.recorded_data_joints = []
        self.recording= True
        self.start_time= time.time()

    def stop_recording_callback(self, msg):
        print "stopping recording ..."
        self.recording= False
        filename = msg.data+str(time.time())+".csv"
        self.print_to_file(filename)

    def auto_save_loop(self, event):
        if self.recording:
            filename = "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/auto_recorded_data/auto_saved" + str(time.time()) + ".csv"

            self.print_to_file(filename)
            print "autosaved to:" +filename


    def recording_loop(self,event):
        if self.recording:
            time_=time.time()-self.start_time
            t = rospy.Time(0)
            try:
                (trans, rot) = self.tf_listener.lookupTransform("/darias", "/human", t)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print "error lookup transform human"
                return

            self.recorded_data_human.append(np.array(trans))

            if self.record_robot:
                try:
                    (trans_end, rot_end) = self.tf_listener.lookupTransform("/darias", "/ENDEFF_RIGHT_ARM", t)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print "error lookup transform endeff "
                    return
                self.recorded_data_endeff.append(np.array(trans_end))
                self.recorded_data_joints.append(self.current_joints)

            self.recorded_times.append(time_)



    def print_to_file(self,filename):
        myFile = open(filename, 'w')
        with myFile:
            if not self.record_robot:
                myFields = ['time_s', 'human_x', 'human_y', 'human_z']
                writer = csv.DictWriter(myFile, fieldnames=myFields)
                writer.writeheader()
                for i in range(len(self.recorded_data_human)):
                    writer.writerow({'time_s': self.recorded_times[i],'human_x':self.recorded_data_human[i][0], 'human_y':self.recorded_data_human[i][1],
                                    'human_z':self.recorded_data_human[i][2]})
            else:
                myFields = ['time_s', 'human_x', 'human_y', 'human_z', 'robot_x','robot_y','robot_z','R_SFE', 'R_SAA', 'R_HR', 'R_EB', 'R_WR','R_WFE','R_WAA' ]
                writer = csv.DictWriter(myFile, fieldnames=myFields)
                writer.writeheader()
                for i in range(len(self.recorded_data_human)):
                    writer.writerow({'time_s': self.recorded_times[i], 'human_x': self.recorded_data_human[i][0],
                                     'human_y': self.recorded_data_human[i][1],
                                     'human_z': self.recorded_data_human[i][2],
                                    'robot_x': self.recorded_data_endeff[i][0],
                                    'robot_y': self.recorded_data_endeff[i][1],
                                    'robot_z': self.recorded_data_endeff[i][2],
                                    'R_SFE': self.recorded_data_joints[i][0],
                                     'R_SAA': self.recorded_data_joints[i][1],
                                     'R_HR': self.recorded_data_joints[i][2],
                                     'R_EB': self.recorded_data_joints[i][3],
                                     'R_WR': self.recorded_data_joints[i][4],
                                     'R_WFE': self.recorded_data_joints[i][5],
                                     'R_WAA': self.recorded_data_joints[i][6]})

        print "saved trajectory to " + filename
        print "start visualizing"
        msg=String()
        msg.data=filename
        self.visualization_publisher.publish(msg)
        #self.visualize2D(filename)

    def visualize2D(self,csv_filename) :
        self.rec_restarted=False
        my_data = genfromtxt(csv_filename, delimiter=',')
        print my_data.shape

        if not self.record_robot:
            f, ax = plt.subplots(1, 1)
            ax.scatter(my_data[1:-1,1],my_data[1:-1,2])
        else:
            f, ax = plt.subplots(1, 1)
            ax.scatter(my_data[1:-1, 2], my_data[1:-1, 1] ,c='b',label='human')

            ax.scatter(my_data[1:-1, 5],my_data[1:-1, 4] ,c='r',label='robot')
            ax.set_xlim([-2.,2.])
            ax.set_ylim([-1., 2.])
            ax.invert_yaxis()

            ax.legend()


        plt.show()

        # while (not self.rec_restarted):
        #     time.sleep(0.1)
        # plt.close('all')
        # print "closed all"














if __name__ == '__main__':
    rospy.init_node("human_trajectory_recording")
    rec = HumanTrajectoryRecording()
    rospy.spin()
