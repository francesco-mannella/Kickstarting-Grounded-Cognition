import sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

def cost_softmax(x, lmb=1):
    e = np.exp(-(x - np.min(x))/lmb)
    return e/sum(e)

def rew_softmax(x, lmb=1):
    e = np.exp((x - np.max(x))/lmb)
    return e/sum(e)

class Env:

    def reset(self):
        pass

    def step(action):
        status = None; reward = None
        return status, reward

class Agent:

    def _init__(self, num_theta, stime, num_inputs, num_outputs):

        self.num_theta = num_theta
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.stime = stime

    def updateTheta(self, theta):
        pass

    def step(self, inputs):
        outputs = None
        return outputs

class CostFuncObject:

    def __init__(self, num_params, stime, env, agent):
        self.env = env
        self.agent = agent

    def compute_inputs(self, status):
        pass

    def onEndEpisode(self):
        pass

    def run_episode(self, save_data):

        stime = self.agent.stime

        rews = []
        action = np.zeros(2)

        data = None
        if save_data is True:
            data = []

        status = self.env.reset()

        for t in range(stime):

            inputs = self.compute_inputs(status)
            action = self.agent.step(inputs)
            status, rew, *_ = self.env.step(action)

            rews.append(rew)

            if save_data is True:
                data.append(status)

        self.onEndEpisode()

        return rews, data

    def __call__(self, thetas, save_data=False):

        rews = []

        if save_data is True:
            data = []

        for theta in thetas:

            self.agent.updateTheta(theta)

            status = self.env.reset()
            self.agent.reset()

            episode_rews, episode_data = self.run_episode(save_data)

            if save_data is True:
                data.append(episode_data)

            rews.append(episode_rews)
        rews = np.vstack(rews)

        if save_data is True:
            return rews, data

        return rews

class BBO :

    "P^2BB: Policy Improvement through Black Vox Optimization"
    def __init__(self,
            cost_func,
            num_params=10,
            num_rollouts=20,
            A=0,
            lmb=0.1,
            epochs=100,
            sigma=0.001,
            sigma_decay_amp=0,
            sigma_decay_period=0.1,
            softmax=rew_softmax):
        '''
        :param num_params: Integer. Number of parameters to optimize
        :param num_rollouts: Integer. number of rollouts per iteration
        :param lmb: Float. Temperature of the evaluation softmax
        :param epochs: Integer. Number of iterations
        :param sigma: Float. Amount of exploration around the mean of parameters
        :param sigma_decay_amp: Initial additive amplitude of exploration
        :param sigma_decay_period: Decaying period of additive
            amplitude of exploration
        '''

        self.sigma = sigma
        self.lmb = lmb
        self.num_rollouts = num_rollouts
        self.num_params = num_params
        self.theta = A + 0.01*np.random.randn(self.num_params)
        self.Cov = np.eye(self.num_params, self.num_params)
        self.epochs = epochs
        self.decay_amp = sigma_decay_amp
        self.decay_period = sigma_decay_period
        self.epoch = 0

        # define softmax
        self.softmax = softmax
        # define the cost function
        self.cost_func = cost_func

    def sample(self):
        """ Get num_rollouts samples from the current parameters mean
        """

        Sigma = self.sigma + self.decay_amp*np.exp(
            -self.epoch/(self.epochs * self.decay_period))

        # matrix of deviations from the parameters mean
        self.eps = np.random.multivariate_normal(
            np.zeros(self.num_params),
            self.Cov * Sigma, self.num_rollouts)

    def update(self, Sk):
        ''' Update parameters

            :param Sk: array(Float), rollout costs in an iteration
        '''
        # Cost-related probabilities of sampled parameters
        probs = self.softmax(Sk, self.lmb).reshape(self.num_rollouts, 1)
        # update with the weighted average of sampled parameters
        self.theta += np.sum(self.eps * probs, 0)


    def outcomes(self):
        """
        compute outcomes for all agents

        :param thetas: array(num_agents X num_params/num_agents)
        """
        thetas = self.theta + self.explore*self.eps
        errs = self.cost_func(thetas)
        return errs

    def eval(self, errs):
        """ evaluate rollouts
            :param errs: Matrix containing agents' errors
                 at each timestep (columns) of each rollout (rows)
            return: array(float), overall cost of each rollout
        """
        timesteps = errs.shape[1]
        # comute costs
        Sk = np.hstack([np.sum([np.sum(err[j:-1])
            for j in range(timesteps)]) for err in errs])

        return Sk

    def iteration(self, explore = True):
        """ Run an iteration
            :param explore: Bool, If the iteration is for training (True)
                or test (False)
            :return: total value of the iteration
        """
        self.explore = explore
        self.sample()
        errs = self.outcomes()
        Sk = self.eval(errs)
        self.update(Sk)
        self.epoch += 1
        return Sk
