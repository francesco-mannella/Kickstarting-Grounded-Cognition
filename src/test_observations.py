import numpy as np
import matplotlib.pyplot as plt

def forwardKinematics(angles, lengths=None, start=None, mirror=False,angle_start=None):

    angles = np.copy(angles)

    if lengths is None:
        lengths = np.ones(len(angles))
    if start is None:
        start = np.zeros(2)
    sum = lambda x: np.sum(x)
    poses = [start]

    if angle_start is None:
        angle_start = 0
   
    for i, length in enumerate(lengths):
        angle = -(angle_start + sum(angles[:i+1]))
        print(angle)
        poses.append(poses[-1] + length*np.array([np.sin(angle), np.cos(angle)]))
    return np.array(poses)

def arm_poses(angles):
    angles = np.copy(angles)
    angles *= [1,1,1,-1,1,1,-1]
    arm_pose = forwardKinematics(angles[:3], lengths=[6,6,6]) 
    grip1_pose = forwardKinematics(angles[3:5], lengths=[3,3], start=arm_pose[-1], angle_start=np.sum(angles[:3]))
    grip2_pose = forwardKinematics(-angles[5:], lengths=[3,3], start=arm_pose[-1], angle_start=np.sum(angles[:3]))
    return arm_pose, grip1_pose, grip2_pose

def grip_poses(angles):
    angles = np.copy(angles)
    angles *= [1,1,1,-1,1,1,-1]
    grip1_pose = forwardKinematics(angles[3:5], lengths=[3,3])
    grip2_pose = forwardKinematics(-angles[5:], lengths=[3,3])
    return grip1_pose, grip2_pose

data = np.load("observations.npy", allow_pickle=True)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, aspect="equal")
grip1, = plt.plot(0,0,c="k",lw=5)
grip2, = plt.plot(0,0,c="k",lw=5)
arm, = plt.plot(0,0,c="k",lw=5)
points = plt.scatter(0,0, c="k",s=200)
ax.set_xlim([-7,7])
ax.set_ylim([-.5,7])
for t in range(1000):
    arm_pose, grip1_pose, grip2_pose = arm_poses(data[t,1])
    grip1_pose, grip2_pose = grip_poses(data[t,1])

    # arm.set_data(*arm_pose.T)
    grip1.set_data(*grip1_pose.T)
    grip2.set_data(*grip2_pose.T)
    points.set_offsets(np.vstack([grip1_pose, grip2_pose]))
    plt.pause(0.01)
