import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import glob
import pickle

goals_data = np.genfromtxt("../evaluation_data_ral/goals/true_goals.csv", dtype=float, delimiter=",", skip_header=1)

true_goals = goals_data[:,0:3]
length_true_goals = goals_data[:,3:5]


fig,ax = plt.subplots(1,1)



idx = 0
dist = np.zeros((25,4))
for filepath in glob.iglob("../evaluation_data_ral/goals/user_goals/*.csv"):

    goals_cur = np.genfromtxt(filepath, dtype=float, delimiter=",", skip_header=1)[:,0:3]

    for i in range(goals_cur.shape[0]):
        ax.scatter(goals_cur[i, 0], goals_cur[i, 1],c ='b')

        #find goal 0

        idx_matching_curr= np.argmin(np.sum((goals_cur[:,:]-true_goals[i,:])**2,axis=1))
         #= []

        dist[idx,i] = np.sqrt(np.sum((goals_cur[idx_matching_curr,:]-true_goals[i,:])**2))
    idx += 1


for i in range(true_goals.shape[0]):
    ax.scatter(true_goals[i,0],true_goals[i,1],c='r')
    ax.add_patch(
        patches.Rectangle(
            (true_goals[i,0]-length_true_goals[i,0]/2., true_goals[i,1]-length_true_goals[i,1]/2.),
            length_true_goals[i, 0],
            length_true_goals[i, 1],
            fill=False  # remove background
        ))



ax.set_xlim([0.,2.0])
ax.set_ylim([-0.8,0.5])


print np.mean(dist,axis=0)
print np.std(dist,axis=0)


with open('../evaluation_data_ral/goals/tmp_all_goals.pkl', 'rb') as f:
    data = pickle.load(f)

all_goals=data["all_goals"]


colors=['r','g','b','m']
plot_idces = [0,1,3,5,10,20,30,49]
all_points=[]
for j in range(len(all_goals)):
    points = all_goals[j][2]
    all_points.append(points)
    if j in plot_idces:
        fig, ax = plt.subplots(1, 1)
        plt.tick_params(labelsize=15)
        current_goals=all_goals[j][0]
        current_cov = all_goals[j][1]

        for i in range(len(all_points)):

            ax.scatter(all_points[i][ 0], all_points[i][ 1],c = 'k')

        for i in range(current_goals.shape[0]):
            ax.scatter(current_goals[i,0],current_goals[i,1],c=colors[i])
            ax.add_patch(patches.Ellipse(xy=[current_goals[i,0],current_goals[i,1]], width=4*np.sqrt(np.diag(current_cov[i,:,:])[0]), height=4*np.sqrt(np.diag(current_cov[i,:,:])[1]),alpha=0.1,color=colors[i]))


        for i in range(true_goals.shape[0]):
            #ax.scatter(true_goals[i,0],true_goals[i,1],c='r')
            ax.add_patch(
                patches.Rectangle(
                    (true_goals[i,0]-length_true_goals[i,0]/2., true_goals[i,1]-length_true_goals[i,1]/2.),
                    length_true_goals[i, 0],
                    length_true_goals[i, 1],
                    fill=False  # remove background
                ))
        ax.set_xlim([0.3, 1.85])
        ax.set_ylim([-0.6, 0.4])
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_xlabel("x [m]",fontsize=15)
        ax.set_ylabel("y [m]",fontsize=15)
        plt.tight_layout()
        fig.savefig("../evaluation_data_ral/goals_"+str(j)+".svg",dpi=fig.dpi)
        print "saved"
        plt.show()

print "bla"