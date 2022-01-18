import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mkvideo import vidManager 

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
    angles *= [-1,1]
    grip1_pose = forwardKinematics(angles, lengths=[3,3])
    grip2_pose = forwardKinematics(-angles, lengths=[3,3])
    return grip1_pose, grip2_pose

plt.ion()

fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(111, aspect="equal")
im1 = ax1.imshow(np.zeros([2,2]))
ax1.set_axis_off()
fig2 = plt.figure(figsize=(10, 10))
ax2 = fig2.add_subplot(111, aspect="equal")
im2 = ax2.imshow(np.zeros([2,2]))
ax2.set_axis_off()
fig3 = plt.figure(figsize=(10, 10))
ax3 = fig3.add_subplot(111, aspect="equal")
im3 = [[[ax3.plot(0,0,lw=3,c='k')[0] for k in range(2)] 
    for x in range(10)] for y in range(10)]
ax3.set_xlim([-7, 2*7*10])
ax3.set_ylim([-7, 2*7*10 - 7])
ax3.set_axis_off()

fig1.tight_layout(0.01)
fig2.tight_layout(0.01)
fig3.tight_layout(0.01)

vm1 = vidManager(fig1, name="map1")
vm2 = vidManager(fig2, name="map2")
vm3 = vidManager(fig3, name="map3")

def scale(x): 
    return (x - x.min())/(x.max() - x.min())

for i in range(80):

    visual_map = np.load("visual_map%06d.npy"%i)
    visual_map = scale((visual_map +1)/2)

    visual_map = visual_map.reshape(10,10,3,10,10)\
            .transpose(3,4,0,1,2)\
            .transpose(0,2,1,3,4)\
            .reshape(100,100,3)
    
    im1.set_array(visual_map)

    visual_map_all = np.load("visual_map_all%06d.npy"%i)
    visual_map_all = scale((visual_map_all +1)/2)

    visual_map_all = visual_map_all.reshape(10,10,3,10,10)\
            .transpose(3,4,0,1,2)\
            .transpose(0,2,1,3,4)\
            .reshape(100,100,3)

    im2.set_array(visual_map_all)

    angles_map = np.load("angles_map%06d.npy"%i)
    angles_map = angles_map.reshape(2,10,10)\
            .transpose(1,2,0)

    for x in range(10):
        for y in range(10):
            p1, p2 = grip_poses(angles_map[x,y]) 

            im3[x][y][0].set_data(*(p1 + [15, 14.8]*np.array([y,x])+ [0.5, -6]).T)
            im3[x][y][1].set_data(*(p2 + [15, 14.8]*np.array([y,x]) +[0.5, -6]).T)

    vm1.save_frame()
    vm2.save_frame()
    vm3.save_frame()


vm1.mk_video()
vm2.mk_video()
vm3.mk_video()

