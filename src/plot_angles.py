import numpy as np
import matplotlib.pyplot as plt

def foorwardKinematics(angles, lengths, start=None, mirror=False):
    if start is None:
        start = np.zeros(2)

    sum = lambda x: np.sum(x)
    if mirror is True:
        sum = lambda x: np.pi-np.sum(x)


    poses = [start]
    for i, length in enumerate(lengths):
        angle = sum(angles[:(i+1)])
        poses.append(poses[-1] + length*np.array([np.cos(angle), np.sin(angle)]))
    return poses

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([-2,2])
ax.set_ylim([-.2,2.2])
f1, = ax.plot(0,0,c="black", lw=2)
f2, = ax.plot(0,0,c="black", lw=2)
for t in range(100):
    angles = np.random.uniform(0, np.pi/2, 2)*[1,1]
    angles[1] = np.minimum(np.pi/2 - angles[0], angles[1])
    p1 = foorwardKinematics(angles, lengths=[1, 1])
    p2 = foorwardKinematics(angles, lengths=[1, 1], mirror=True)
    f1.set_data(*np.array(p1).T)
    f2.set_data(*np.array(p2).T)
    plt.pause(0.1)
