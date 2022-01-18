import sys, os
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
import params
from bbo import BBO
import gym, box2dsim
from mkvideo import vidManager

from ArmActuator  import ArmActuator, Agent
from ArmActuatorDevel import Env

class Objective:

    def __init__(self, *args, timesteps=100, contexts=5, **kargs):
        self.env = Env(*args, **kargs)
        self.agent = Agent(env=self.env, *args, **kargs)
        self.timesteps = timesteps
        self.contexts = contexts
        self.episode_count = 0

    def getReward(self):
        dist = np.linalg.norm(self.state-self.goal)
        s = 2
        return np.exp(-0.5*(s**-2)*dist**2)

    def episode(self, context):

        vm = vidManager(fig=None, name="frame", dirname="frames", duration=30)

        self.env.render = "offline"
        self.env.b2d_env.render_init(self.env.render)

        reward = np.zeros(self.timesteps)
        self.goal = self.agent.arm.interpolate(context)
        self.goal += np.random.randn(*self.goal.shape)
        self.state = self.env.reset()
        self.agent.reset()
        axis = self.env.b2d_env.renderer.ax
        if axis is not None:
            axis.scatter(*self.goal, s=20, c="black")
        for t in range(self.timesteps):
            action = self.agent.step(self.goal)
            self.state = self.env.step(np.hstack([action, np.zeros(2)]))
            reward[t] = self.getReward()

        print("episode{}".format(self.episode_count))
        vm.mk_video(dir=".", name="episode{}".format(self.episode_count))

        self.env.render = "none"
        self.env.b2d_env.render_init(self.env.render)
        plt.close("all")

        return reward

    def __call__(self, params):
        rewards = []
        for p,param in enumerate(params):
            self.agent.updatePolicy(param)
            self.episode_count = 0
            context_rewards = []
            for c in range(self.contexts):
                curr_goal = c/self.contexts + 0.001*np.random.randn()
                reward = self.episode(curr_goal)
                context_rewards.append(reward)
                self.episode_count += 1
            context_rewards = np.vstack(context_rewards)
            rewards.append(context_rewards.mean(0))
        rewards = np.vstack(rewards)
        return rewards

def test():
    env = gym.make('Box2DSimOneArmOneEye-v0')
    num_inputs = 2
    num_hidden = 600
    num_outputs = 3

    bbo_num_params = num_hidden*num_outputs
    bbo_num_rollouts = 1
    bbo_lmb = 0.0001
    bbo_epochs = 1
    timesteps = 100
    bbo_sigma = 0.0001
    bbo_sigma_decay_amp = 0.0
    bbo_sigma_decay_period = 99999

    objective = Objective(
        contexts=20,
        box2d_env=env,
        timesteps=timesteps,
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        actuator_map_name="data/StoredArmActuatorMap",
        actuator_weights_name="data/StoredArmActuatorWeights")

    bbo = BBO(num_params=bbo_num_params,
            num_rollouts=bbo_num_rollouts,
            lmb=bbo_lmb,
            epochs=bbo_epochs,
            sigma=bbo_sigma,
            sigma_decay_amp=bbo_sigma_decay_amp,
            sigma_decay_period=bbo_sigma_decay_period,
            cost_func=objective)

    bbo.theta = np.copy(objective.agent.arm.params)

    for epoch in range(bbo_epochs):
        e = bbo.iteration(explore=False)
        print("{: 4d}".format(epoch))
        return bbo

if __name__ == "__main__":
    bbo = test()
