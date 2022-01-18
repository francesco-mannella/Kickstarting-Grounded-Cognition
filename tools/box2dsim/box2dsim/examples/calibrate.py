import numpy as np
import gym
import box2dsim

rng = np.random.RandomState(62)
env = gym.make('Box2DSimOneArmOneEye-v0')
env.set_world(3)
stime = 10000
action = [0, 0, 0, np.pi*0.3, np.pi*0.3]
for t in range(stime):
    env.render()
    if t < stime/2:
        action += 0*np.pi*rng.randn(5)
        action[3:] = 0
        action[:3] = np.maximum(-np.pi, np.minimum(0, action[:3]))
        print(action)
        env.step(action)
