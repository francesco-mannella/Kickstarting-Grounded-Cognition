from SMController import SMController
from SMEnv import SMEnv
from SMAgent import SMAgent
import params
import numpy as np
from glob import glob
import regex

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



class SensoryMotorCicle:

    def __init__(self):
        self.t = 0

    def step(self, env, agent, state, action_steps = 10):
        if self.t % action_steps == 0:
            self.action = agent.step(state)
        state = env.step(self.action)
        self.t += 1
        return state

seed = 5563
rng = np.random.RandomState(seed)
env = SMEnv(seed)
agent = SMAgent(env)
controller = SMController(rng)

internal_side = int(np.sqrt(params.internal_size))
visual_side = int(np.sqrt(params.visual_size / 3))

epoch=int(regex.sub(".*_(\d+).npy", "\\1", sorted(glob("storage/visual_weight*.npy"))[-1]))
print(epoch)

weights = {
        "visual": np.load(f"storage/visual_weights_{epoch:04d}.npy"),
        "ssensory": np.load(f"storage/ssensory_weights_{epoch:04d}.npy"),
        "proprio": np.load(f"storage/proprio_weights_{epoch:04d}.npy"),
        "policy": np.load(f"storage/policy_weights_{epoch:04d}.npy"),
        "predict": np.load(f"storage/comp_weights_{epoch:04d}.npy"),
        }
controller.load(weights)


CONTEXTS = {    
    "unreachable": 0,    
    "still": 1,    
    "movable": 2,    
    "controllable": 3,    
    "noobject": 4,    
}   

state = env.reset(CONTEXTS["controllable"], plot="test", render="offline")
agent.reset()
                
v = state["VISUAL_SENSORS"].ravel()
ss = state["TOUCH_SENSORS"]
p = state["JOINT_POSITIONS"][:5]
a = np.zeros(agent.params_size)

internal_representations,_ = controller.spread([[v], [ss], [p], [a]])

# take only vision
internal_mean = internal_representations[0]
policy = controller.getPoliciesFromRepresentations(internal_mean)
agent.updatePolicy(policy)


import matplotlib.pyplot as plt
from mkvideo import vidManager

saliency_fig = plt.figure(figsize=env.b2d_env.renderer_figsize)
saliency_ax =  saliency_fig.add_subplot(111)
saliency_img = saliency_ax.imshow(
        np.zeros([
            env.b2d_env.bground_pixel_side,
            env.b2d_env.bground_pixel_side,
            ]), 
        vmin=0, 
        vmax=1,
        cmap=plt.cm.Greys,
        )

retina_fig = plt.figure(figsize=env.b2d_env.renderer_figsize)
retina_ax =  retina_fig.add_subplot(111)
retina_img = retina_ax.imshow(
        np.zeros([
            env.b2d_env.fovea_pixel_side,
            env.b2d_env.fovea_pixel_side,
            3]), 
        )

gretina_fig = plt.figure(figsize=env.b2d_env.renderer_figsize)
gretina_ax =  gretina_fig.add_subplot(111)
gretina_img = gretina_ax.imshow(
        np.zeros([
            env.b2d_env.fovea_pixel_side,
            env.b2d_env.fovea_pixel_side,
            3]), 
        )

smv = vidManager(saliency_fig, "saliency", "saliency_frames")
rmv = vidManager(retina_fig, "retina", "retina_frames")
grmv = vidManager(gretina_fig, "gretina", "gretina_frames")

smcycle = SensoryMotorCicle()
for t in range(params.stime):
    state = smcycle.step(env, agent, state)

    v = state["VISUAL_SENSORS"]
    sal = state["VISUAL_SALIENCY"]

    sal /= sal.max()
    saliency_img.set_array(sal)
    retina_img.set_array(v)
    smv.save_frame()
    rmv.save_frame()
    

    o_r = controller.stm_v.spread([v.ravel()])
    p = controller.stm_v.getPoint(o_r)
    v_r = controller.stm_v.getRepresentation(p, 0.8)
    v_g = controller.stm_v.backward([v_r]).numpy()
    gretina_img.set_array(np.maximum(0, v_g.reshape(10, 10, 3)/v_g.max()))

    grmv.save_frame()


env.close()

wfile=f"storage/visual_weights_{epoch:04d}.npy"
data_v = np.load(wfile)
data_v = data_v.reshape(visual_side, visual_side, 3, internal_side, internal_side)
data_v = data_v.transpose(3, 0, 4, 1, 2)
data_v = data_v.reshape(visual_side * internal_side, visual_side * internal_side, 3)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, aspect="equal")
ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
ax.set_axis_off()
fig.tight_layout(pad=0.0)
fig.savefig("visual_map.png")
