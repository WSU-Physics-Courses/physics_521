from __future__ import division
from collections import namedtuple

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint
from IPython.display import display
import sympy
from sympy import symbols, trigsimp, Eq, S


def get_xy(theta1, theta2, r1, r2, h=0, np=np):
    """Return (xs, ys) with the positions of the
    fulcrum and two masses.  h is the height of the pivot
    point in case you want to drive the system.
    """
    x1 = r1*np.sin(theta1)
    y1 = -r1*np.cos(theta1) + h
    x2 = x1 + r2*np.sin(theta2)
    y2 = y1 - r2*np.cos(theta2)
    return [0, x1, x2], [h, y1, y2]


class DrawPendulum(object):
    """This class provides some methods for drawing the pendulum system."""
    def __init__(self, r1, r2, hmax=0,
                 fig=None, ax=None, figsize=(10, 5)):
        if fig is None:
            fig = plt.figure(figsize=figsize)
        self.fig = fig
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.r1 = r1
        self.r2 = r2

        self._line = self.draw_pendulum()

        R = r1+r2
        self.ax.set_xlim(-1.1*R, 1.1*R)
        self.ax.set_ylim(-1.1*(R+hmax), 1.1*(R+hmax))
        self.ax.set_aspect(1)

    def draw_pendulum(self, theta1=0, theta2=0, h=0,
                      **kw):
        args = dict(color='k', ls='-', lw=5, ms=20)
        args.update(kw)

        xs, ys = get_xy(theta1, theta2, self.r1, self.r2, h)
        return self.ax.plot(xs, ys, 'o', **args)[0]

    def update_pendulum(self, theta1=0, theta2=0, h=0):
        """Update the position of the pendulum"""
        xs, ys = get_xy(theta1, theta2, self.r1, self.r2, h)
        self._line.set_data(xs, ys)

    def text(self, s, theta1, r1, theta2=0, r2=0, **kw):
        [_, _, x], [_, _, y] = get_xy(theta1, theta2, r1, r2)
        self.ax.text(x, y, s, ha='center', va='center', **kw)


def draw_figure():
    th1 = 0.7
    th2 = 1.7
    r1 = 1.0
    r2 = 2.0
    p = DrawPendulum(r1, r2)
    p.update_pendulum(th1, th2)
    p.draw_pendulum(0, 0, ls=':', lw=1, ms=0)
    p.draw_pendulum(th1, 0, ls=':', lw=1, ms=0)
    p.text(r'$\theta_1$', theta1=th1/2, r1=r1/2)
    p.text(r'$\theta_2$', theta1=th1, r1=r1, theta2=th2/2, r2=r1/2)
    p.text(r'$r_1$', theta1=th1+0.5, r1=r1/1.6)
    p.text(r'$r_2$', theta1=th1, r1=r1, theta2=th2+0.2, r2=r2/2)
    p.text(r'$m_1$', theta1=th1, r1=r1, theta2=-th1, r2=r1/2)
    p.text(r'$m_2$', theta1=th1, r1=r1, theta2=th2-0.2, r2=r2)


def get_eqs():
    from . import euler_lagrange

    th1, th2 = symbols(r'\theta_1, \theta_2', real=True, cls=sympy.Function)

    t, g, r1, r2, m1, m2 = symbols('t, g, r_1, r_2, m_1, m_2', positive=True)

    ms = [m1, m2]
    rs = [r1, r2]

    ths = [th1(t), th2(t)]
    dths = [_f.diff(t) for _f in ths]
    ddths = [_df.diff(t) for _df in dths]
    funcs = ths
    dfuncs = [_f.diff(t) for _f in funcs]
    ddfuncs = [_df.diff(t) for _df in dfuncs]

    # These are substitutions to make things look nicer by replacing
    # derivatives with dots.
    vsubs = []
    vsubs.extend((_d, symbols(r'\ddot{%s}' % (_d.expr.func,))) for _d in ddfuncs)
    vsubs.extend((_d, symbols(r'\dot{%s}' % (_d.expr.func,))) for _d in dfuncs)
    vsubs.extend((_d, symbols(str(_d.func))) for _d in funcs)
    ddth1, ddth2 = symbols(r'\ddot{\theta}_1, \ddot{\theta}_2')

    # Here we express x and y in terms of the thetas
    (_, x1, x2), (_, y1, y2) = get_xy(*(ths + rs), np=sympy)

    K = trigsimp(
        m1/2*(x1.diff(t)**2 + y1.diff(t)**2)
        + m2/2*(x2.diff(t)**2 + y2.diff(t)**2)
        )
    V = trigsimp(g*(m1*y1 + m2*y2))
    L = K - V

    # This generates numerical version of the EL equations for use later.
    # There is some magic here you can ask me about if you are interested.
    res = euler_lagrange.get_rhs(L, funcs=ths, var=t, simplify=True)
    args = [g, m1, m2, r1, r2] + ths + dths
    _ddth1 = euler_lagrange.my_lambdify(args, res[ddths[0]])
    _ddth2 = euler_lagrange.my_lambdify(args, res[ddths[1]])

    L = L.subs(vsubs)
    th1, th2 = symbols([r'\theta_1', r'\theta_2'])
    dth1, dth2 = symbols([r'\dot{\theta_1}', r'\dot{\theta_2}'])


    def get_solution(th1, dth1, th2, dth2, T,
                     g=9.81, m1=1.0, m2=1.0, r1=1.0, r2=1.0):
        """Return the solution given the specified initial conditions and
        parameters for 3 periods T."""

        x0 = [th1, dth1, th2, dth2]
        Nt = 1000
        ts = np.linspace(0, 3*T, Nt)

        def f(x, t):
            th1, dth1, th2, dth2 = x
            dth1_dt = dth1
            dth2_dt = dth2
            ddth1_dt = _ddth1(g, m1, m2, r1, r2, th1, th2, dth1, dth2)
            ddth2_dt = _ddth2(g, m1, m2, r1, r2, th1, th2, dth1, dth2)
            return dth1_dt, ddth1_dt, dth2_dt, ddth2_dt

        th1, dth1, th2, dth2 = odeint(f, x0, ts).T

        Solution = namedtuple('Solution', ['t', 'theta_1', 'theta_2'])
        return Solution(t=ts, theta_1=th1, theta_2=th2)
    return locals()
