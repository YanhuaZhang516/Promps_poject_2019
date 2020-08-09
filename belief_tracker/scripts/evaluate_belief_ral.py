import rospy
import numpy as np
import matplotlib.pyplot as plt

import  belief_tracker.belief_tracker_class as belief_tracker_class
import csv

subject='C17W'
# Initialize the belief tracker
goals_file = "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/recorded_data/userstudy_all/"+subject+"/goals_"+subject+".csv"
parameter_file = "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/recorded_data/userstudy_all/"+subject+"/belief_"+subject+".pkl"
csv_file= "/home/doro/ias_ros/src/pomdp_collision/belief_tracker/recorded_data/userstudy_all/"+subject+"/"+subject+"_no_robot.csv"
bt = belief_tracker_class.BeliefTracker(use_ros=False)


step = 3
start=480 #100
end=600
predictions,beliefs,avg_vels,at_goals, radius_around_goal, vel_history,stay_dur = bt.sim_belief_from_csv_file(csv_file,start,end,step,goals_file,parameter_file)

with open(csv_file, "r") as csv_file:
    trajectory = []
    times = []
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for row in csv_reader:
        trajectory.append([float(row["human_x"]), float(row["human_y"]), float(row["human_z"])])
        times.append(float(row['time_s']))

goals = np.genfromtxt(goals_file, dtype=float, delimiter=",",skip_header=1)[:,0:3]


fig, ax = plt.subplots(1, 1)
plt.tick_params(labelsize=15)
human_all=np.asarray(trajectory)
ax.plot(human_all[:,0],human_all[:,1])
ax.set_xlim([0.3, 1.85])
ax.set_ylim([-0.6, 0.4])
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_xlabel("x [m]",fontsize=15)
ax.set_ylabel("y [m]",fontsize=15)
plt.tight_layout()
plt.savefig("../eval_belief/data_human.svg", dpi=fig.dpi)

font = {'family': 'serif',
        'color':  'blue',
        'weight': 'bold',
        'size': 14,
        }
offset_text = np.array([[-0.1,-0.1],[-0.1,-0.1],[-0.1,-0.1],[-0.1,-0.1]])
for i in range(2,len(predictions)):
    fig, ax = plt.subplots(1, 1)
    trajs = predictions[i]
    human_data = np.asarray(trajectory[start:start + (i) * step:step])
    for j in range(len(trajs)):
        traj = trajs[j]
        ax.plot([human_data[-1, 0], traj [0,0]], [human_data[-1, 1],traj[0,1]],'g',alpha=0.5)
        ax.plot(traj[:,0],traj[:,1],'g',linewidth=4, alpha=0.5)
        ax.scatter(traj[:, 0], traj[:, 1], c='g',s=5)


    ax.plot (human_data[:,0],human_data[:,1],c='y',linewidth=3, alpha=0.2)


    for g in range(goals.shape[0]):
        circle = plt.Circle((goals[g,0],goals[g,1]), radius_around_goal[g], color='b', fill=False,alpha=0.5,linewidth=4)
        ax.add_artist(circle)
        ax.text(goals[g,0]+offset_text[g,0],goals[g,1]+offset_text[g,1], b'G'+str(g), fontdict=font)
        #ax.scatter(goals[g,0],goals[g,1],c='g')
    ax.scatter(human_data[-1, 0], human_data[-1, 1],marker='*' ,c='k',zorder=10,s=160)
    ax.set_xlim([0.3, 1.85])
    ax.set_ylim([-0.6, 0.4])
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("x [m]", fontsize=15)
    ax.set_ylabel("y [m]", fontsize=15)
    title_belief = "b(g): ["
    for b in range(len(beliefs[i])):
        title_belief+="{:10.4f}".format(beliefs[i][b])

    title_belief+="]"# vel:"+"{:10.2f}".format(avg_vels[i])+"  atgoal:"+str(at_goals[i])#+"\n stay:"+str(stay_dur[i])
    print title_belief
    ax.set_title(title_belief,fontsize=15)
    plt.tight_layout()
    plt.savefig("../eval_belief/data_"+str(i)+".svg", dpi=fig.dpi)
plt.show()
print "init done"