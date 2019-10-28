"""VPython visualization of a rotating gyroscope sitting on a pedestal.

Modified from the sources here:

http://www.glowscript.org/#/user/GlowScriptDemos/folder/Examples/
"""
import math

import time

import numpy as np
from scipy.integrate import odeint

import numpy as np
import vpython as vp


from utils import NoInterrupt, Frame


class PendulumVisualization(object):
    g = 9.81                    # Gravity
    m_1 = m_2 = 1.0             # Mases
    L_1 = L_2 = 1.0             # Lengths

    dt = 0.05                    # Steps between frames

    rod_thickness = 0.01
    ball_radius = 0.1

    # Initial conditions
    theta1 = 0.1
    theta2 = 0.1
    theta1dot = 0.0
    theta2dot = 0.0
    retain = 100

    def __init__(self, Pendulum, theta1=0.0, theta2=0.0, epsilon=0.001):
        self.Pendulum = Pendulum
        self.theta1 = theta1
        self.theta2 = theta2
        self.epsilon = epsilon

        scene = vp.canvas()
        scene.width = 800
        scene.height = 600
        scene.range = 2.2
        scene.up = vp.vec(0, 1, 0)
        scene.center = vp.vec(0, 0, 0)
        scene.forward = vp.vec(0, 0, -1)

        scene.userzoom = False
        scene.title = "A double pendulum"

        # frame = Frame(size=0.4)

        #rod1 = vp.cylinder(pos=vp.vec(0, 0, 0),
        #                   axis=vp.vec(self.L_1, 0, 0),
        #                   radius=self.tshaft,
        #                   color=vp.color.orange)
        kw = dict(radius=self.rod_thickness)
        rod1 = vp.cylinder(pos=vp.vec(0, 0, 0),
                           axis=vp.vec(0, -1, 0), **kw)
        rod2 = vp.cylinder(pos=vp.vec(0, -1, 0),
                           axis=vp.vec(0, -1, 0), **kw)
        rod1e = vp.cylinder(pos=vp.vec(0, 0, 0),
                            axis=vp.vec(0, -1, 0), **kw)
        rod2e = vp.cylinder(pos=vp.vec(0, -1, 0),
                            axis=vp.vec(0, -1, 0), **kw)
        kw = dict(radius=self.ball_radius)
        pivot = vp.sphere(pos=vp.vec(0, 0, 0),
                          color=vp.color.yellow, **kw)
        ball1 = vp.sphere(pos=vp.vec(0, -1, 0),
                          color=vp.color.red,
                          make_trail=True,
                          retain=self.retain, **kw)
        ball2 = vp.sphere(pos=vp.vec(0, -2, 0),
                          color=vp.color.blue,
                          make_trail=True,
                          retain=self.retain, **kw)
        ball1e = vp.sphere(pos=vp.vec(0, -1, 0),
                           color=vp.color.red,
                           make_trail=True,
                           retain=self.retain, **kw)
        ball2e = vp.sphere(pos=vp.vec(0, -2, 0),
                           color=vp.color.blue,
                           make_trail=True,
                           retain=self.retain, **kw)
        # tip.trail_color = vp.color.red
        # tip.trail_radius = 0.15*shaft.radius

        self.scene = scene
        self.rods = [(rod1, rod2),
                     (rod1e, rod2e)]
        self.balls = [(ball1, ball2),
                      (ball1e, ball2e)]

        self.reset()
        # self.scene.waitfor('textures')
        self.scene.waitfor('draw_complete')

    def reset(self, random=False):
        q0 = (self.theta1, self.theta2, self.theta1dot, self.theta2dot)
        p0 = self.Pendulum(q0)
        q1 = (self.theta1+self.epsilon, self.theta2, self.theta1dot, self.theta2dot)
        p1 = self.Pendulum(q1)
        self.pendulums = [p0, p1]
        self.clear_trails()
        self.update()
        self.t = 0.0

    def clear_trails(self):
        for (b1, b2) in self.balls:
            b1.clear_trail()
            b2.clear_trail()

    def update(self):
        for p, (b1, b2), (r1, r2) in zip(self.pendulums, self.balls, self.rods):
            [(x1, y1), (x2, y2)] = p.get_xy(p.q)
            b1.pos = vp.vec(x1, y1, 0)
            b2.pos = vp.vec(x2, y2, 0)

            r1.axis = b1.pos
            r2.pos = b1.pos
            r2.axis = b2.pos - b1.pos

    def run(self, duration=60.0):
        raw_input("press any key to start")
        t0 = time.time()
        self.reset()
        init = True
        with NoInterrupt(ignore=True) as interrupted:
            while not interrupted and time.time() < t0 + duration:
                vp.rate(50)
                for p in self.pendulums:
                    p.step(self.dt)
                if init:
                    # Clear trail on first step.
                    self.clear_trails()
                    init = False
                self.update()


class DoublePendulum(object):
    g = 1.0
    m_1 = m_2 = 1.0
    L_1 = L_2 = 1.0

    def __init__(self, q0, t0=0.0):
        self.q = q0
        self.t = t0
        self._qs = [q0]
        self.ts = [t0]

    @property
    def qs(self):
        return np.asarray(self._qs).T

    @property
    def energy(self):
        th1, th2, dth1, dth2 = self.q
        M = self.m_1 + self.m_2
        m = self.m_2
        L1, L2 = self.L_1, self.L_2
        E = (M*dth1**2*L1**2/2.0 + m*dth2**2*L2**2/2.0
             + m*dth1*dth2*L1*L2*math.cos(th1-th2)
             - self.g*(M*L2*math.cos(th1) + m*L2*math.cos(th2)))
        return E

    def get_xy(self, qs=None):
        """Return [(x1, y1), (x2, y2)] for the pendulum nodes."""
        if qs is None:
            qs = self.qs
        th1, th2 = qs[:2]
        r1 = -1j*self.L_1 * np.exp(1j*th1)
        r2 = r1 -1j*self.L_2 * np.exp(1j*th2)
        return [(r1.real, r1.imag),
                (r2.real, r2.imag)]

    def rhs(self, q, t, np=math):
        g, m_1, m_2, L_1, L_2 = self.g, self.m_1, self.m_2, self.L_1, self.L_2
        M = m_1 + m_2
        sin, cos = np.sin, np.cos
        th1, th2, dth1, dth2 = q
        ddth1 = (2*g*m_1*sin(th1)
                 + g*m_2*(sin(th1 - 2*th2) + sin(th1))
                 + m_2*L_1*sin(2*th1 - 2*th2)*dth1**2
                 + 2*m_2*L_2*sin(th1-th2)*dth2**2
                 )/(2*L_1*(m_2*cos(th1-th2)**2 - M))

        ddth2 = (M*(g*sin(th2) - L_1*sin(th1 - th2)*dth1**2)
                 - (g*M*sin(th1) + m_2*L_2*sin(th1-th2)*dth2**2)*cos(th1-th2)
                 )/(L_2*(m_2*cos(th1-th2)**2 - M))
        return dth1, dth2, ddth1, ddth2

    def step(self, dt):
        """Evolve the system by `dt`, saving the results"""
        self.q = odeint(self.rhs, self.q, [self.t, self.t+dt])[-1]
        self.t += dt
        self._qs.append(self.q)
        self.ts.append(self.t)

    def plot(self, **kw):
        [(x1s, y1s), (x2s, y2s)] = self.get_xy()
        plt.plot(x1s, y1s, 'r', **kw)
        plt.plot(x2s, y2s, 'b', **kw)
        plt.gca().set_aspect(1)
