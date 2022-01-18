import glob
import os,sys

from SMGraphs import remove_figs, blank_video, visual_map, comp_map, trajectories_map, representations_movements, log
from SMController import SMController
from SMEnv import SMEnv
from SMAgent import SMAgent
import params
import numpy as np
import matplotlib
import time
import tensorflow as tf


matplotlib.use("Agg")

np.set_printoptions(formatter={"float": "{:6.4f}".format})

class TimeLimitsException(Exception):
    pass

class SensoryMotorCicle:

    def __init__(self):
        self.t = 0

    def step(self, env, agent, state, action_steps = 5):
        if self.t % action_steps == 0:
            self.action = agent.step(state)
        state = env.step(self.action)
        self.t += 1
        return state

class Main:

    def __init__(self, seed=None, plots=False):

        self.plots = plots
        if seed is None:
            seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(seed)
        tf.random.set_seed(seed)

        self.start =  time.perf_counter()
        if self.plots is True: remove_figs()
        self.env = SMEnv(seed)
        self.agent = SMAgent(self.env)
        self.controller = SMController(self.rng)
        self.logs = np.zeros([params.epochs, 3])
        self.epoch = 0


    def train(self, time_limits):


        env = self.env
        agent = self.agent
        controller = self.controller
        logs = self.logs

        batch_data_size = params.batch_size * (params.stime + 1)

        epoch = self.epoch

        time_start = time.perf_counter()
        while epoch < params.epochs:

            total_time_elapsed = time.perf_counter() - self.start
            if total_time_elapsed >= time_limits:
                if self.epoch > 0:
                    raise TimeLimitsException

            print(f"{epoch:6d}", end=" ")

            controller.comp_grid = controller.getCompetenceGrid()
            comp = controller.comp_grid.mean()
            contexts = self.rng.randint(0, params.num_objects, params.batch_size)

            side = np.sqrt(params.internal_size)
            b = params.base_match_sigma
            controller.match_sigma = params.match_sigma * (b + (1 - b) * (1 - comp))
            b = params.base_internal_sigma
            controller.curr_sigma = side * (b + (1 - b) * np.exp(-0.5*(0.1**-2)*((1-comp)-0.3)**2))
            controller.explore_sigma = params.explore_sigma 
            b = params.base_lr
            curr_lr = params.stm_lr * (b + (1 - b) * (1 - comp))
            
            controller.updateParams(controller.curr_sigma, curr_lr)
            policies, competences, goals = controller.getPoliciesRandomly()

            batch_v = np.zeros([batch_data_size, params.visual_size])
            batch_ss = np.zeros([batch_data_size, params.somatosensory_size])
            batch_p = np.zeros([batch_data_size, params.proprioception_size])
            batch_a = np.zeros([batch_data_size, params.policy_size])
            batch_c = np.ones([batch_data_size, 1])
            batch_g = np.zeros([batch_data_size, params.internal_size])
            batch_n = np.zeros(batch_data_size)

            # run episodes
            comps = batch_c
            stime = params.stime + 1
            for episode in range(params.batch_size):

                i = episode * stime
                state = env.reset(contexts[episode])

                batch_v[i, :] = state["VISUAL_SENSORS"].ravel()
                batch_ss[i, :] = state["TOUCH_SENSORS"]
                batch_p[i, :] = state["JOINT_POSITIONS"][:5]

                agent.updatePolicy(policies[episode])
                
                
                batch_a[i:(i+stime), :] = policies[episode]
                batch_g[i:(i+stime), :] = goals[episode]
                batch_c[i:(i+stime), :] = competences[episode]

                smcycle = SensoryMotorCicle()
                for t in range(params.stime):
                    
                    state = smcycle.step(env, agent, state)

                    it = i + t + 1
                    batch_v[it, :] = state["VISUAL_SENSORS"].ravel()
                    batch_ss[it, :] = state["TOUCH_SENSORS"]
                    batch_p[it, :] = state["JOINT_POSITIONS"][:5]

            
            (v_r, ss_r, p_r, a_r),(v_p, ss_p, p_p, a_p) = \
                    controller.spread([batch_v, batch_ss, batch_p, batch_a])
            match, match_increment = controller.computeMatch(np.stack([v_p, ss_p, p_p]), a_p)

            update_items = controller.update(
                batch_v,
                batch_ss,
                batch_p,
                batch_a,
                batch_g,
                match.reshape(-1, 1),
                100*match_increment.reshape(-1, 1),
                competences=batch_c,
            )


            print(f"{update_items:#7d}", end=' ')
            print(f"{batch_ss.sum():#10.2f}", end=' ')
            logs[epoch] = [competences.min(), competences.mean(), competences.max()]
            print(("%16.15f " * 3) % (competences.min(), competences.mean(), competences.max()))

            # diagnose
            if (epoch > 0 and epoch % params.epochs_to_test == 0) or epoch == (params.epochs - 1):

                data = {}
                data["match"] = match
                data["match_increment"] = match_increment
                data["v_r"] = v_r
                data["ss_r"] = ss_r
                data["p_r"] = p_r
                data["a_r"] = a_r
                data["v"] = batch_v
                data["ss"] = batch_ss
                data["p"] = batch_p
                data["a"] = batch_a

                np.save(f"epoch_{epoch:06d}", [data])
                self.diagnose()

                time_elapsed = time.perf_counter() - time_start
                print("---- TIME: %10.4f" % time_elapsed)
                time_start = time.perf_counter()

            epoch += 1
            self.epoch = epoch
            sys.stdout.flush()


    def diagnose(self):
        
        env = self.env
        agent = self.agent
        controller = self.controller
        logs = self.logs
        epoch = self.epoch

        controller.save(epoch)
        if self.plots is False: return

        print("----> Graphs  ...")
        remove_figs(epoch)
        np.save("log", logs[: epoch + 1])
        visual_map()

        log()
        comp_map()
    
        
        if os.path.isfile("PLOT_SIMS"): 
            print("----> Test Sims ...", end = " ", flush=True)
            
            for k in range(params.tests):
            
                context = k % 4
                state = env.reset(context, plot="episode%d" % k, render="offline")
                agent.reset()
                                
                v = state["VISUAL_SENSORS"].ravel()
                ss = state["TOUCH_SENSORS"]
                p = state["JOINT_POSITIONS"][:5]
                a = np.zeros(agent.params_size)

                internal_representations,_ = controller.spread([[v], [ss], [p], [a]])

                # take only vision
                internal_mean = internal_representations[0]
                policy = controller.getPoliciesFromRepresentationsWithNoise(internal_mean)
                agent.updatePolicy(policy)
            
                smcycle = SensoryMotorCicle()
                for t in range(params.stime):
                    state = smcycle.step(env, agent, state)
            
                env.close()
                if k % 2 == 0 or k == params.tests - 1:
                    print("{:d}% ".format(int(100*(k/(params.tests-1)))), end = " ",  flush=True)
            print()
            

        if os.path.isfile("COMPUTE_TRAJECTORIES"): 
            print("----> Compute Trajectories ...", end = " ", flush=True)
            context = 4 # no object
            trj = np.zeros([params.internal_size, params.stime, 2])

            state = env.reset(context)
            agent.reset()
            for i, goal_r in enumerate(controller.goal_grid):
                policy = controller.getPoliciesFromRepresentations([goal_r])
                agent.updatePolicy(policy)
                smcycle = SensoryMotorCicle()
                for t in range(params.stime):
                    state = smcycle.step(env, agent, state)
                    trj[i, t] =state["JOINT_POSITIONS"][-2:] 
                if i % 10 == 0 or i == params.internal_size -1:
                    print("{:d}% ".format(int(100*(i/params.internal_size))), end = " ",  flush=True)
            print()
            np.save("trajectories", trj)
            trajectories_map()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('-t','--time',
            help="The maximum time for the simulation (seconds)",
            action="store", default=1e99) 
    parser.add_argument('-g','--gpu',
            help="Use gpu",
            action="store_true") 
    parser.add_argument('-s','--seed',
            help="Simulation seed",
            action="store", default=1) 
    parser.add_argument('-x','--plots',
            help="Plot graphs",
            action="store_true") 
    args = parser.parse_args()
    timing = float(args.time)
    gpu = bool(args.gpu)
    seed = int(args.seed)
    plots = bool(args.plots)
    
    if not gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    main = Main(seed, plots)
    try:
        main.train(timing)
    except TimeLimitsException:
        print(f"Epoch {main.epoch}. end")
        main.diagnose()  
