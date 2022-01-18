import os
import params
import numpy as np
from SMSTM import SMSTM
from SMPredict import SMPredict
import tensorflow as tf

def softmax(x, lmb=1):
    e = np.exp((x - np.max(x))/lmb)
    return e/sum(e)

class SMController:
    def __init__(self, rng=None):

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

        self.stm_v = SMSTM(
            inp_num=params.visual_size,
            out_num=params.internal_size,
            sigma=params.internal_sigma,
            name="stm_v",
            lr=params.stm_lr,
        )
        self.stm_ss = SMSTM(
            inp_num=params.somatosensory_size,
            out_num=params.internal_size,
            sigma=params.internal_sigma,
            name="stm_ss",
            lr=params.stm_lr,
        )
        self.stm_p = SMSTM(
            inp_num=params.proprioception_size,
            out_num=params.internal_size,
            sigma=params.internal_sigma,
            name="stm_p",
            lr=params.stm_lr,
        )
        self.stm_a = SMSTM(
            inp_num=params.policy_size,
            out_num=params.internal_size,
            sigma=params.internal_sigma,
            name="stm_a",
            lr=params.stm_lr,
        )
        self.predict = SMPredict(
            params.internal_size, 1.0, name="predict", lr=params.predict_lr
        )

        self.match_sigma = params.match_sigma
        self.sigma = params.internal_sigma
        self.curr_sigma = self.sigma
        self.comp_sigma = params.base_internal_sigma
        self.explore_sigma = params.explore_sigma

        self.internal_side = int(np.sqrt(params.internal_size))
        x = np.arange(self.internal_side)
        self.radial_grid = np.stack(np.meshgrid(x, x)).reshape(2, -1).T
        x = self.radial_grid
        self.goal_grid = self.goals_real = np.exp(
            -0.5
            * (self.comp_sigma ** -2)
            * np.linalg.norm(x.reshape(-1, 1, 2) - x.reshape(1, -1, 2), axis=2) ** 2
        )
        self.getCompetenceGrid()

    def getPoliciesFromRepresentations(self, representations):
        return self.stm_a.backward(representations)
    
    def getPoliciesFromPoints(self, points):
        representations = self.stm_a.getRepresentation(points)
        policies = self.getPoliciesFromRepresentations(representations)
        return policies, representations

    def getPoliciesFromRepresentationsWithNoise(self, representations):
        policies = self.getPoliciesFromRepresentations(representations)
        competences = self.predict.spread(representations)
        policies =  policies * (competences + \
                (1 - competences))*(self.explore_sigma*self.rng.randn(*policies.shape))
        return policies

    def getPoliciesRandomly(self):
        goals = self.rng.uniform(0, self.internal_side, [params.batch_size, 2])
        policies, goals_r = self.getPoliciesFromPoints(goals)
        competences = self.predict.spread(goals_r)
        policies =  policies * (competences + \
                (1 - competences))*(self.explore_sigma*self.rng.randn(*policies.shape))

        return policies, competences, goals_r

    def computeMatch(self, representations, target):

        diffs = np.zeros([target.shape[0], len(representations)])
        for i, r in enumerate(representations):
            diffs[:, i] = np.linalg.norm(r -  target, axis=1) / self.match_sigma
            diffs[:, i] == np.exp(-0.5 * (self.match_sigma ** -2) * (diffs[:, i]) ** 2)

        def get_incr(x):
            y = x.reshape(params.batch_size, -1)
            return np.hstack([np.zeros([params.batch_size, 1]), 
                    np.maximum(0,np.diff(y))]) 
        matches_increments = np.stack([get_incr(diffs[:, x]).ravel() for x in range(3)])
        matches_increments = matches_increments.transpose(1, 0)
        matches_weights = np.eye(3)[matches_increments.argmax(1)]

        matches = np.mean(diffs * matches_weights, axis=1)
        matches_increments = np.mean(matches_increments * matches_weights, axis=1)
        
        return matches, matches_increments.ravel()

    def getCompetenceGrid(self):
        g = self.predict.spread(self.goal_grid)
        return g

    def spread(self, inps):
        v, ss, p, a = inps
        v_out = self.stm_v.spread(v)
        ss_out = self.stm_ss.spread(ss)
        p_out = self.stm_p.spread(p)
        a_out = self.stm_a.spread(a)
        
        v_p = self.stm_v.getPoint(v_out)
        ss_p = self.stm_ss.getPoint(ss_out)
        p_p = self.stm_p.getPoint(p_out)
        a_p = self.stm_a.getPoint(a_out)

        v_r = self.stm_v.getRepresentation(v_p, params.base_internal_sigma)
        ss_r = self.stm_ss.getRepresentation(ss_p, params.base_internal_sigma)
        p_r = self.stm_p.getRepresentation(p_p, params.base_internal_sigma)
        a_r = self.stm_a.getRepresentation(a_p, params.base_internal_sigma)

        return (v_r, ss_r, p_r, a_r), (v_p, ss_p, p_p, a_p)

    def updateParams(self, sigma, lr):
        self.stm_v.updateParams(sigma=sigma, lr=lr)
        self.stm_ss.updateParams(sigma=sigma, lr=lr)
        self.stm_p.updateParams(sigma=sigma, lr=lr)
        self.stm_a.updateParams(sigma=sigma, lr=lr)
        self.sigma = sigma

    def update(
        self,
        visuals,
        ssensories,
        proprios,
        policies,
        goals,
        matches,
        matches_increment,
        competences,
    ):

        cgoals = goals * (1 - competences)
        
        mx = matches.max()
        if mx <= 0: mx = 1
        mch_idcs = matches.ravel()/mx > params.match_th 
        mx = matches_increment.max()
        if mx <= 0: mx = 1
        mch_idcs &= matches_increment.ravel()/mx > params.match_incr_th 
        mch_idcs &= ssensories.sum(1) > 0
        n_items = sum(1*mch_idcs)
        mx = matches.max()

        self.stm_v.update(visuals[mch_idcs], cgoals[mch_idcs] * matches[mch_idcs]/mx)
        self.stm_ss.update(ssensories[mch_idcs], cgoals[mch_idcs] * matches[mch_idcs]/mx)
        self.stm_p.update(proprios[mch_idcs], cgoals[mch_idcs] * matches[mch_idcs]/mx)
        self.stm_a.update(policies[mch_idcs], cgoals[mch_idcs] * matches[mch_idcs]/mx)
        self.predict.update(goals, matches)

        return n_items

    def save(self, epoch):
        os.makedirs("storage", exist_ok=True)
        np.save(f"visual_weights", self.stm_v.weights())
        np.save(f"comp_grid", self.comp_grid)
        np.save(f"storage/visual_weights_{epoch:04d}", self.stm_v.weights())
        np.save(f"storage/ssensory_weights_{epoch:04d}", self.stm_ss.weights())
        np.save(f"storage/proprio_weights_{epoch:04d}", self.stm_p.weights())
        np.save(f"storage/policy_weights_{epoch:04d}", self.stm_a.weights())
        np.save(f"storage/comp_grid_{epoch:04d}", self.comp_grid)
        np.save(f"storage/comp_weights_{epoch:04d}", self.predict.weights())

    def load(self, weights):
        self.stm_v.get_weights(weights["visual"])
        self.stm_ss.get_weights(weights["ssensory"])
        self.stm_p.get_weights(weights["proprio"])
        self.stm_a.get_weights(weights["policy"])
        self.predict.get_weights(weights["predict"])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    smcontrol = SMController()

