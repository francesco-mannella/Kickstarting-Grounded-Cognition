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
from ArmActuator import Agent

class Env:

    def __init__(self, box2d_env, **kargs):
        self.b2d_env = box2d_env
        self.b2d_env.set_taskspace(**params.task_space)
        self.render = None
        self.rng = self.b2d_env.rng
        self.reset()

    def step(self, action):
        self.b2d_env.step(action)
        if self.render is not None:
            self.b2d_env.render(self.render)
        return self.b2d_env.handPosInSpace()

    def reset(self):
        self.b2d_env.set_world(self.b2d_env.worlds["noobject"])
        if self.render is not None:
            self.b2d_env.render(self.render)
        return self.b2d_env.handPosInSpace()

class Objective:

    def __init__(self, *args, contexts=5, **kargs):
        self.env = Env(*args, **kargs)
        self.agent = Agent(env=self.env, *args, **kargs)
        self.timesteps = params.stime
        self.contexts = contexts
        self.episode_count = 0
        self.plot = False

    def getReward(self):
        dist = np.linalg.norm(self.state-self.goal)
        s = 2
        return np.exp(-0.5*(s**-2)*dist**2)

    def episode(self, context):
        reward = np.zeros(self.timesteps)
        self.goal = self.agent.arm.interpolate(context) + self.env.rng.randn(2)
        self.state = self.env.reset()
        self.agent.reset()

        for t in range(self.timesteps):
            if t==0: action = self.agent.step(self.goal)
            self.state = self.env.step(np.hstack([action, np.zeros(2)]))
            reward[t] = self.getReward()

        return reward

    def episode_plot(self, context):

        vm = vidManager(fig=None, name="frame", dirname="frames",
            duration=30)
        
        reward = np.zeros(self.timesteps)
        self.goal = self.agent.arm.interpolate(context)
        self.state = self.env.reset()
        self.agent.reset()
        
        self.env.render = "offline"
        self.env.b2d_env.render_init(self.env.render)
        axis = self.env.b2d_env.renderer.ax
        axis.scatter(*self.goal, s=20, c="black")
        
        for t in range(self.timesteps):
            if t==0: action = self.agent.step(self.goal)
            action = self.agent.step(self.goal)
            self.state = self.env.step(np.hstack([action, np.zeros(2)]))
            reward[t] = self.getReward()
        
        print(".", end="")
        vm.mk_video(dir=".",
            name="episode{}".format(self.episode_count))
        self.env.render = None
        self.env.b2d_env.render_init(self.env.render)
        plt.close("all")
        
        return reward
        

    def __call__(self, params):

        rewards = np.zeros([len(params), self.timesteps])
        for p, param in enumerate(params):
            self.agent.updatePolicy(param)
            self.episode_count = 0
            context_rewards = np.zeros([self.contexts, self.timesteps])
            for i,c in enumerate(range(self.contexts)):
                curr_goal = (c/self.contexts)
                if self.plot is True:
                    reward = self.episode_plot(curr_goal)
                else:
                    reward = self.episode(curr_goal)

                context_rewards[i] = reward
                self.episode_count += 1

            rewards[p] = context_rewards.mean(0)

        return rewards

def train():
    env = gym.make('Box2DSimOneArmOneEye-v0')
    num_inputs = 2
    num_hidden = 10*10
    num_outputs = 3

    bbo_num_params = num_hidden*num_outputs
    bbo_num_rollouts = 10
    bbo_lmb = 0.0001
    bbo_epochs = 1000
    bbo_sigma = 0.001
    bbo_A = 1.2

    bbo_sigma_decay_amp = 0.0
    bbo_sigma_decay_period = 99999

    epochs_to_plot = 5

    objective = Objective(
        contexts=7,
        box2d_env=env,
        num_inputs=num_inputs,
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        actuator_map_name="data/StoredArmActuatorMap")

    bbo = BBO(num_params=bbo_num_params,
            num_rollouts=bbo_num_rollouts,
            lmb=bbo_lmb,
            A=bbo_A, 
            epochs=bbo_epochs,
            sigma=bbo_sigma,
            sigma_decay_amp=bbo_sigma_decay_amp,
            sigma_decay_period=bbo_sigma_decay_period,
            cost_func=objective)

    logs = np.zeros([bbo_epochs, 3])
    for epoch in range(bbo_epochs):
        e = bbo.iteration()
        print("{: 4d} {:10.8f}".format(epoch, np.mean(e)))
        logs[epoch] = [np.min(e), np.mean(e), np.max(e)]

        if (epoch % epochs_to_plot) == 0:
            print("Plotting", end="")
            objective.plot = True
            objective([bbo.theta])
            objective.plot = False
            print(" done.")

    bbo.logs = logs
    weights = bbo.theta
    np.save("data/StoredArmActuatorWeights",  weights)
    
    return bbo

if __name__ == "__main__":
    bbo = train()
