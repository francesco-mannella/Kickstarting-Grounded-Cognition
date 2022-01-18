import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
import params
from bbo import BBO
from esn import ESN
import gym, box2dsim
from mkvideo import vidManager
from GripMapping import Env as GripEnv
from GripActuator import Agent as GripAgent
from ArmActuator import Agent as ArmAgent

class DevEnv(GripEnv):

    def reset(self, world=None):
        state = super(DevEnv, self).reset(world)
        if self.render == "offline":
            self.vm = vidManager(fig=None, name="frame", dirname="frames",
                duration=30)
        else:
            self.vm = None
        return state

    def saveVideo(self, count):
        if self.vm is not None:
            self.vm.mk_video(dir=".",
                name="episode{}".format(count))



class Objective:

    def __init__(self, b2d_env, *args, contexts=5, **kargs):
        arm_input = kargs["arm_input"]
        arm_hidden = kargs["arm_hidden"]
        arm_output = kargs["arm_output"]
        grip_input = kargs["grip_input"]
        grip_hidden = kargs["grip_hidden"]
        grip_output = kargs["grip_output"]
        self.env = DevEnv(b2d_env)
        self.arm_agent = ArmAgent(env=self.env, 
                num_inputs=arm_input, num_hidden=arm_hidden, num_outputs=arm_output,
                actuator_map_name="data/StoredArmActuatorMap",
                actuator_weights_name="data/StoredArmActuatorWeights", 
                *args, **kargs)
        self.grip_agent = GripAgent(env=self.env, 
                num_inputs=grip_input, num_hidden=grip_hidden, num_outputs=grip_output,
                actuator_map_name="data/StoredGripActuatorMap",
                actuator_weights_name="data/StoredGridActuatorWeights", 
                *args, **kargs)
        self.grip_param_shape = self.grip_agent.grip.params.shape
        self.params_size = np.prod(self.grip_param_shape) 
        self.timesteps = params.stime
        self.contexts = contexts
        self.episode_count = 0
        self.plot = False

    def getReward(self):
        rew = np.sum(self.state["TOUCH_SENSORS"]**2)
        return rew

    def episode(self, context, render=None):
        self.env.render = render

        reward = np.zeros(self.timesteps)
        self.state = self.env.reset(context)
        self.arm_agent.reset()
        self.grip_agent.reset()

        for t in range(self.timesteps):
            pos = self.state["EYE_POS"][::-1]
            touch = self.state["TOUCH_SENSORS"] 
            joints = self.state["JOINT_POSITIONS"][3:5] 
            arm_state = self.state["EYE_POS"][::-1]
            grip_state = np.hstack([pos, touch, joints])
            arm_action = self.arm_agent.step(arm_state)
            grip_action = self.grip_agent.step(grip_state)
            action = np.hstack([
                arm_action + grip_action[:3]*0.4, 
                grip_action[3:]])
            self.state = self.env.step(action)
            reward[t] = self.getReward()
        
        return reward

    def __call__(self, params):
        rewards = []
    
        for p,param in enumerate(params):
            self.grip_agent.updatePolicy(param)
            self.episode_count = 0
            context_rewards = []
            for c in range(self.contexts):
                if c%4 == 0: 
                    world = c//4
                reward = self.episode(world, render = "offline" 
                        if self.plot is True else None)
                context_rewards.append(reward)
                if self.plot:
                    self.env.saveVideo(self.episode_count)
                self.episode_count += 1
            context_rewards = np.vstack(context_rewards)
            rewards.append(context_rewards.mean(0))
        rewards = np.vstack(rewards)
        return rewards

def train():

    env = gym.make('Box2DSimOneArmOneEye-v0')
    
    bbo_num_params = 500
    bbo_num_rollouts = 10
    bbo_lmb = 0.0001
    bbo_epochs = 1000
    bbo_sigma = 0.1
    contexts = 16

    bbo_sigma_decay_amp = 0.0
    bbo_sigma_decay_period = 99999

    epochs_to_plot = 5
    
    objective = Objective(env, 
            contexts=contexts, 
            arm_input=2,
            arm_hidden=100,
            arm_output=3,
            grip_input=6,
            grip_hidden=100,
            grip_output=5)
    
    bbo = BBO(num_params=bbo_num_params,
            num_rollouts=bbo_num_rollouts,
            A=0.2,
            lmb=bbo_lmb,
            epochs=bbo_epochs,
            sigma=bbo_sigma,
            sigma_decay_amp=bbo_sigma_decay_amp,
            sigma_decay_period=bbo_sigma_decay_period,
            cost_func=objective)

    logs = np.zeros([bbo_epochs, 3])
    for epoch in range(bbo_epochs):
        e = bbo.iteration()
        print("{: 4d} {:10.2f}".format(epoch, np.mean(e)))
        logs[epoch] = [np.min(e), np.mean(e), np.max(e)]

        if (epoch % epochs_to_plot) == 0:
            print("Plotting", end="")
            objective.plot = True
            objective([bbo.theta])
            objective.plot = False
            print(" done.")

    bbo.logs = logs
    weights = bbo.theta
    np.save("data/StoredGripActuatorWeights",  weights)
    
    return bbo
if __name__ == "__main__":
   train() 
