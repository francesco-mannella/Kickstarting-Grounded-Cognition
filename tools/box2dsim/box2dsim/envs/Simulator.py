from . import JsonToPyBox2D as json2d
from .PID import PID
import time, sys, os, glob

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class  Box2DSim(object):
    """ 2D physics using box2d and a json conf file
    """
    @staticmethod
    def loadWorldJson(world_file):
        jsw = json2d.load_json_data(world_file)
        return jsw


    def __init__(self, world_file=None, world_dict=None, dt=1/80.0, vel_iters=30, pos_iters=2):
        """
        Args:

            world_file (string): the json file from which all objects are created
            world_dict (dict): the json object from which all objects are created
            dt (float): the amount of time to simulate, this should not vary.
            pos_iters (int): for the velocity constraint solver.
            vel_iters (int): for the position constraint solver.

        """
        if world_file is not None:
            world, bodies, joints = json2d.createWorldFromJson(world_file)
        else:
            world, bodies, joints = json2d.createWorldFromJsonObj(world_dict)
        self.dt = dt
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters
        self.world = world
        self.bodies = bodies
        self.joints = joints
        self.joint_pids = { ("%s" % k): PID(dt=self.dt)
                for k in list(self.joints.keys()) }

    def contacts(self, bodyA, bodyB):
        """ Read contacts between two parts of the simulation

        Args:

            bodyA (string): the name of the object A
            bodyB (string): the name of the object B

        Returns:

            (int): number of contacts
        """

        contacts = 0
        for ce in self.bodies[bodyA].contacts:
            if ce.contact.touching is True:
                if ce.contact.fixtureB.body  == self.bodies[bodyB]:
                    contacts += len(ce.contact.manifold.points)
        return contacts

    def move(self, joint_name, angle):
        """ change the angle of a joint

        Args:

            joint_name (string): the name of the joint to move
            angle (float): the new angle position

        """
        pid = self.joint_pids[joint_name]
        pid.setpoint = angle

    def step(self):
        """ A simulation step
        """
        for key in list(self.joints.keys()):
            self.joint_pids[key].step(self.joints[key].angle)
            self.joints[key].motorSpeed = (self.joint_pids[key].output)
        self.world.Step(self.dt, self.vel_iters, self.pos_iters)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from matplotlib.path import Path

class VisualSensor:
    """ Compute the retina state at each ste of simulation
    """

    def __init__(self, sim, shape, rng):
        """
        Args:

            sim (Box2DSim): a simulator object
            shape (int, int): width, height of the retina in pixels
            rng (float, float): x and y range in the task space

        """

        self.shape = list(shape)
        self.n_pixels = self.shape[0] * self.shape[1]

        # make a canvas with coordinates
        x = np.arange(-self.shape[0]//2, self.shape[0]//2) + 1
        y = np.arange(-self.shape[1]//2, self.shape[1]//2) + 1
        X, Y = np.meshgrid(x, y[::-1])
        self.grid = np.vstack((X.flatten(), Y.flatten())).T
        self.scale = np.array(rng)/shape
        self.radius = np.mean(np.array(rng)/shape)
        self.retina = np.zeros(self.shape + [3])

        self.reset(sim);

    def reset(self, sim):
        self.sim = sim

    def step(self, focus) :
        """ Run a single simulator step

        Args:

            sim (Box2DSim): a simulator object
            focus (float, float): x, y of visual field center

        Returns:

            (np.ndarray): a rescaled retina state
        """

        self.retina *= 0
        for key in self.sim.bodies.keys():
            body = self.sim.bodies[key]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            vercs = vercs[np.arange(len(vercs))+[0]]
            data = [body.GetWorldPoint(vercs[x])
                for x in range(len(vercs))]
            body_pixels =  self.path2pixels(data, focus)
            if body.color is None: body.color = [0.5, 0.5, 0.5]
            color = np.array(body.color)
            body_pixels = body_pixels.reshape(body_pixels.shape + (1,))*(1 - color)
            self.retina += body_pixels
        self.retina = np.maximum(0, 1 - (self.retina))
        return self.retina

    def path2pixels(self, vertices, focus):

        points = self.grid * self.scale + focus

        path = Path(vertices) # make a polygon
        points_in_path = path.contains_points(points, radius=self.radius)
        img = 1.0*points_in_path.reshape(*self.shape, order='F').T #pixels

        return img
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class TestPlotter:
    """ Plotter of simulations
    Builds a simple matplotlib graphic environment
    and render single steps of the simulation within it

    """

    def __init__(self, env, xlim=[-10, 30], ylim=[-10, 30], figsize=None, offline=False):
        """
        Args:
            env (Box2DSim): a emulator object

        """

        self.env = env
        self.offline = offline
        self.xlim = xlim
        self.ylim = ylim

        if figsize is None:
            self.fig = plt.figure()
        else:
            self.fig = plt.figure(figsize=figsize)

        self.ax = None

        self.reset()

    def close(self):
        plt.close(self.fig)

    def reset(self):

        if self.ax is not None:
            plt.delaxes(self.ax)
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.polygons = {}
        for key in self.env.sim.bodies.keys() :
            self.polygons[key] = Polygon([[0, 0]],
                    ec=self.env.sim.bodies[key].color + [1],
                    fc=self.env.sim.bodies[key].color + [1],
                    closed=True)

            self.ax.add_artist(self.polygons[key])

        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        if not self.offline:
            self.fig.show()
        else:
            self.ts = 0


    def onStep(self):
        pass

    def step(self) :
        """ Run a single emulator step
        """

        for key in self.polygons:
            body = self.env.sim.bodies[key]
            vercs = np.vstack(body.fixtures[0].shape.vertices)
            data = np.vstack([ body.GetWorldPoint(vercs[x])
                for x in range(len(vercs))])
            self.polygons[key].set_xy(data)

        self.onStep()

        if not self.offline:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw()
        else:
            if not os.path.exists("frames"):
                os.makedirs("frames")

            self.fig.savefig("frames/frame%06d.png" % self.ts, dpi=200)
            self.fig.canvas.draw()
            self.ts += 1

class TestPlotterOneEye(TestPlotter):
    def __init__(self, *args, **kargs):

        super(TestPlotterOneEye, self).__init__(*args, **kargs)
        self.eye_pos, = self.ax.plot(0, 0)

    def reset(self):
        super(TestPlotterOneEye, self).reset()
        self.eye_pos, = self.ax.plot(0, 0)

    def onStep(self):

        pos = np.copy(self.env.eye_pos)
        x = pos[0] + np.array([-1, -1, 1,  1, -1])*self.env.fovea_height*0.5
        y = pos[1] + np.array([-1,  1, 1, -1, -1])*self.env.fovea_width*0.5
        self.eye_pos.set_data(x, y)
