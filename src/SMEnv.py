import numpy as np
import params
import gym, box2dsim
from mkvideo import vidManager

class SMEnv:

    def __init__(self, seed):
        self.b2d_env = gym.make('Box2DSimOneArmOneEye-v0')
        self.b2d_env = self.b2d_env.unwrapped
        self.b2d_env.set_seed(seed)

        self.b2d_env.set_taskspace(**params.task_space)
        self.render = None
        self.world = 0 

    def step(self, action):
        observation,*_ = self.b2d_env.step(action)
        if self.render is not None:
            self.b2d_env.render(self.render)
        return observation

    def reset(self, world=None, plot=None, render=None):
        self.render = render
        self.plot = plot
        if world is not None:
            self.world = world

        observation = self.b2d_env.reset(self.world)
        if self.render is not None:
            if self.plot is not None:
                self.vm = vidManager(fig=None, 
                        name="frame", duration=30)
            self.b2d_env.render_init(self.render)

        return observation

    def close(self):
        if self.plot is not None:
            self.vm.mk_video(name=self.plot, dirname=".")
            self.vm.clear() 

