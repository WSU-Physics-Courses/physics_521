"""Various utilities tools for VPython etc.
"""
from collections import namedtuple
import signal
import time

from threading import RLock

import numpy as np

import vpython as vp


class NoInterrupt(object):
    """Suspend the various signals during the execution block.

    Arguments
    ---------
    ignore : bool
       If True, then do not raise a KeyboardInterrupt if a soft interrupt is
       caught.

    Note: This is not yet threadsafe.  Semaphores should be used so that the
      ultimate KeyboardInterrupt is raised only by the outer-most context (in
      the main thread?)  The present code works for a single thread because the
      outermost context will return last.

      See:

      * http://stackoverflow.com/questions/323972/
        is-there-any-way-to-kill-a-thread-in-python

    >>> import os, signal, time

    This loop will get interrupted in the middle so that m and n will not be
    the same.

    >>> def f(n, interrupted=False, force=False):
    ...     done = False
    ...     while not done and not interrupted:
    ...         n[0] += 1
    ...         if n[0] == 5:
    ...             # Simulate user interrupt
    ...             os.kill(os.getpid(), signal.SIGINT)
    ...             if force:
    ...                 # Simulated a forced interrupt with multiple signals
    ...                 os.kill(os.getpid(), signal.SIGINT)
    ...                 os.kill(os.getpid(), signal.SIGINT)
    ...             time.sleep(0.1)
    ...         n[1] += 1
    ...         done = n[0] >= 10

    >>> n = [0, 0]
    >>> try:  # All doctests need to be wrapped in try blocks to not kill py.test!
    ...     f(n)
    ... except KeyboardInterrupt, err:
    ...     print("KeyboardInterrupt: {}".format(err))
    KeyboardInterrupt:
    >>> n
    [5, 4]

    Now we protect the loop from interrupts.
    >>> n = [0, 0]
    >>> try:
    ...     with NoInterrupt() as interrupted:
    ...         f(n)
    ... except KeyboardInterrupt, err:
    ...     print("KeyboardInterrupt: {}".format(err))
    KeyboardInterrupt:
    >>> n
    [10, 10]

    One can ignore the exception if desired:
    >>> n = [0, 0]
    >>> with NoInterrupt(ignore=True) as interrupted:
    ...     f(n)
    >>> n
    [10, 10]

    Three rapid exceptions will still force an interrupt when it occurs.  This
    might occur at random places in your code, so don't do this unless you
    really need to stop the process.
    >>> n = [0, 0]
    >>> try:
    ...     with NoInterrupt() as interrupted:
    ...         f(n, force=True)
    ... except KeyboardInterrupt, err:
    ...     print("KeyboardInterrupt: {}".format(err))
    KeyboardInterrupt: Interrupt forced
    >>> n
    [5, 4]


    If `f()` is slow, we might want to interrupt it at safe times.  This is
    what the `interrupted` flag is for:

    >>> n = [0, 0]
    >>> try:
    ...     with NoInterrupt() as interrupted:
    ...         f(n, interrupted)
    ... except KeyboardInterrupt, err:
    ...     print("KeyboardInterrupt: {}".format(err))
    KeyboardInterrupt:
    >>> n
    [5, 5]

    Again: the exception can be ignored
    >>> n = [0, 0]
    >>> with NoInterrupt(ignore=True) as interrupted:
    ...     f(n, interrupted)
    >>> n
    [5, 5]
    """
    _instances = set()  # Instances of NoInterrupt suspending signals
    _signals = set((signal.SIGINT, signal.SIGTERM))
    _signal_handlers = {}  # Dictionary of original handlers
    _signals_raised = []
    _all_signals_raised = []
    _force_n = 3

    # Time, in seconds, for which 3 successive interrupts will raise a
    # KeyboardInterrupt
    _force_timeout = 1

    # Lock should be re-entrant (I think) since a signal might be sent during
    # operation of one of the functions.
    _lock = RLock()

    @classmethod
    def catch_signals(cls, signals=None):
        """Set signals and register the signal handler if there are any
        interrupt instances."""
        with cls._lock:
            if signals:
                cls._signals = set(signals)
                cls._reset_handlers()

            if cls._instances:
                # Only set the handlers if there are interrupt instances
                cls._set_handlers()

    @classmethod
    def _set_handlers(cls):
        with cls._lock:
            cls._reset_handlers()
            for _sig in cls._signals:
                cls._signal_handlers[_sig] = signal.signal(
                    _sig, cls.handle_signal)
        
    @classmethod
    def _reset_handlers(cls):
        with cls._lock:
            for _sig in list(cls._signal_handlers):
                signal.signal(_sig, cls._signal_handlers.pop(_sig))

    @classmethod
    def handle_signal(cls, signum, frame):
        with cls._lock:
            cls._signals_raised.append((signum, frame, time.time()))
            cls._all_signals_raised.append((signum, frame, time.time()))
            if cls._forced_interrupt():
                raise KeyboardInterrupt("Interrupt forced")

    @classmethod
    def _forced_interrupt(cls):
        """Return True if `_force_n` interrupts have been recieved in the past
        `_force_timeout` seconds"""
        with cls._lock:
            return (cls._force_n <= len(cls._signals_raised)
                    and
                    cls._force_timeout > (cls._signals_raised[-1][-1] -
                                          cls._signals_raised[-3][-1]))

    def __init__(self, ignore=False):
        self.ignore = ignore
        self.tic = time.time()
        NoInterrupt._instances.add(self)
        self.catch_signals()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            self._instances.remove(self)
            if not self._instances:
                # Only raise an exception if all the instances have been
                # cleared, otherwise we might still be in a protected
                # context somewhere.
                self._reset_handlers()
                if self:
                    # An interrupt was raised.
                    while self._signals_raised:
                        # Clear previous signals
                        self._signals_raised.pop()
                    if exc_type is None and not self.ignore:
                        raise KeyboardInterrupt()

    @classmethod
    def __nonzero__(cls):
        with cls._lock:
            return bool(cls._signals_raised)

    __bool__ = __nonzero__  # For python 3


class Object(object):
    """Wrapper for incomplete object types providing the functions I need."""
    Type = None

    def __init__(self, *v, **kw):
        self.__dict__['_object'] = self.Type(*v, **kw)

    def __getattr__(self, key):
        return getattr(self._object, key)

    def __setattr__(self, key, value):
        setattr(self._object, key, value)


class Collection(Object):
    """Collection of objects that is useful.

    The first object in the list is used as a reference point.
    """
    def __init__(self, objects):
        self.__dict__['_objects'] = objects

    @property
    def _object(self):
        return self._objects[0]

    def rotate(self, angle, axis=None, origin=None):
        origin = self.pos if origin is None else origin
        axis = self.axis if axis is None else axis
        for o in self._objects:
            o.rotate(angle=angle, axis=axis, origin=origin)

    def __setattr__(self, key, value):
        if key == 'pos':
            dpos = value - self.pos
            for o in self._objects:
                o.pos += dpos
        else:
            for o in self._objects:
                setattr(o, key, value)

    def __getitem__(self, key):
        return self._objects[key]

    def __iter__(self):
        return self._objects.__iter__()


class Text(Object):
    Type = vp.label

    def rotate(self, angle, axis, origin=None):
        if origin is None:
            # Nothing to do... label's have no orientation.
            return
        self.pos = origin + (self.pos - origin).rotate(angle=angle, axis=axis)

    def __setattr__(self, key, value):
        if key in ['axis']:
            pass
        else:
            Object.__setattr__(self, key, value)


class Frame(Collection):
    def __init__(self,
                 pos=vp.vec(0, 0, 0),
                 axis=vp.vec(1, 0, 0),
                 up=vp.vec(0, 1, 0),
                 size=1.0,
                 labels=True,
                 shaftwidth=0.05,
                 labelsize=None,
                 ball=0.01, make_trail=True, **kw):
        axis *= size
        shaftwidth *= size
        ball *= size
        x = axis.norm()
        y = (up - up.proj(x)).norm()
        z = x.cross(y)

        # Rotation matrix
        R = np.array([x.value, y.value, z.value]).T
        arrows = []
        for _x in R.T:
            _x = _x*size
            args = dict(pos=pos, axis=vp.vec(*_x), color=vp.vec(*_x),
                        shaftwidth=shaftwidth)
            args.update(kw)
            arrows.append(vp.arrow(**args))

        texts = []
        if labels:
            for _n, f in enumerate(arrows):
                args = dict(text='xyz'[_n], pos=f.pos + f.axis,
                            align='center',  height=labelsize,
                            color=f.color)
                args.update(kw)
                texts.append(Text(**args))

        balls = []
        if ball > 0:
            for f in arrows:
                args = dict(radius=ball, pos=f.pos+f.axis, color=f.color,
                            make_trail=make_trail, retain=30)
                args.update(kw)
                balls.append(vp.sphere(**args))

        self.__dict__.update(_balls=balls, _arrows=arrows, _texts=texts)

    @property
    def _objects(self):
        return self._arrows + self._balls + self._texts

    def reset(self, pos=None, axis=vp.vec(1, 0, 0), up=vp.vec(0, 1, 0),
              phi=None, theta=None, psi=None):
        if pos is None:
            pos = self.pos
        x = axis.norm()
        y = (up - up.proj(x)).norm()
        z = x.cross(y)

        if phi is not None:
            x, y, z = [_x.rotate(angle=phi, axis=z) for _x in [x, y, z]]
        if theta is not None:
            x, y, z = [_x.rotate(angle=theta, axis=x) for _x in [x, y, z]]
        if psi is not None:
            x, y, z = [_x.rotate(angle=psi, axis=z) for _x in [x, y, z]]

        # Rotation matrix
        # R = np.array([x.value, y.value, z.value]).T
        fx, fy, fz = self._objects[:3]
        for _f, _x, _y in zip([fx, fy, fz], [x, y, z], [y, z, x]):
            _f.pos = pos
            _f.axis = _x
            _f.up = _y

        for _t, _f in zip(self._texts, self._arrows):
            _t.pos = _f.pos + _f.axis

        if self._balls:
            for _t, _f in zip(self._balls, self._arrows):
                _t.pos = _f.pos + _f.axis

    @property
    def x(self):
        return self.axis.norm()

    @property
    def y(self):
        return self.up.norm()

    @property
    def z(self):
        return self.x.cross(self.y)


class EulerAngleExplorer(object):
    state = np.array([0.0, 0, 0])
    new_state = np.array([0.0, 0, 0])
    step = 0.1

    def __init__(self):
        self.scene = vp.canvas(forward=vp.vec(-1, -1, -1),
                               up=vp.vec(0, 0, 1),
                               userzoom=False)

        sliders = []
        sliders.append(vp.slider(text='phi', min=0.0, max=2*np.pi, value=self.phi,
                                 bind=self._set_phi))
        self.scene.append_to_caption('phi\n')

        sliders.append(vp.slider(text='theta', min=0.0, max=np.pi, value=self.theta,
                                 bind=self._set_theta))
        self.scene.append_to_caption('theta\n')

        sliders.append(vp.slider(text='psi', min=0.0, max=2*np.pi, value=self.psi,
                                 bind=self._set_psi))
        self.scene.append_to_caption('psi\n')

        self.frame0 = Frame(opacity=0.5, labels=False)
        self.frame = Frame()
        self.sliders = sliders

    @property
    def phi(self):
        return self.state[0]

    @phi.setter
    def phi(self, phi):
        self.state[0] = phi

    @property
    def theta(self):
        return self.state[1]

    @theta.setter
    def theta(self, theta):
        self.state[1] = theta

    @property
    def psi(self):
        return self.state[2]

    @psi.setter
    def psi(self, psi):
        self.state[2] = psi

    def _set_phi(self, s):
        self.new_state[0] = s.value
        self.tic = time.time()

    def _set_theta(self, s):
        self.new_state[1] = s.value
        self.tic = time.time()

    def _set_psi(self, s):
        self.new_state[2] = s.value
        self.tic = time.time()

    def run(self, timeout=30):
        self.tic = time.time()
        with NoInterrupt(ignore=True) as interrupted:
            while not interrupted and time.time() < self.tic + timeout:
                vp.rate(10)
                self.update()

    def update(self):
        f = self.frame
        if np.allclose(self.new_state, self.state):
            vp.sleep(0.1)
            return

        ds = (self.new_state - self.state)
        n = 1.0 + int(np.linalg.norm(ds)/self.step)
        ds /= n
        self.state += ds
        f.reset(phi=self.phi, theta=self.theta, psi=self.psi)

    def __del__(self):
        del self.frame
        del self.frame0


class Point(object):
    """Represents a point of an object, providing visualizations for the force,
    velocity, etc."""

    _vectors = ['x', 'v', 'F']  # Vector properties
    _defaults = dict(radius=0.05)

    def __init__(self, x=np.zeros(3), v=None, m=None, F=None, **kw):
        x = np.asarray(x)
        if v is None:
            v = np.zeros_like(x)
        if F is None:
            F = np.zeros_like(x)
        if m is None:
            m = 1.0
        self.x = x
        self.m = m
        self.v = v
        self.F = F
        self._kw = dict(self._defaults, **kw)

    def __repr__(self):
        args = ", ".join(["{}={}".format(_k, self.__dict__[_k])
                          for _k in list(self.__dict__) + list(self._kw)
                          if not _k.startswith('_')])
        rep = "{}({})".format(self.__class__.__name__, args)
        return rep

    def __getattr__(self, key):
        """Allows for accessing attributes as vp.vec by appending _vec."""
        if key.endswith('_vec'):
            # Convert to a vp.vec
            value = list(getattr(self, key[:-4]))
            value += [0]*(3-len(value))
            return vp.vec(*value)
        return self.__getattribute__(key)

    def __setattr__(self, key, value):
        """Makes sure attributes are arrays."""
        if key in self._vectors:
            value = np.asarray(value)
        object.__setattr__(self, key, value)
        self._update()

    def _update(self):
        """Update visualization."""
        if hasattr(self, '_visualization'):
            objects = self._visualization._objects
            objects.x.pos = objects.F.pos = self.x_vec
            objects.F.axis = self.F_vec

    def get_visualization(self, forces=True, **kw):
        kw = dict(self._kw, **kw)
        x = vp.sphere(pos=self.x_vec, **kw)
        F = vp.arrow(pos=self.x_vec, axis=self.F_vec, visible=forces, **kw)
        objects = namedtuple('PointVisualization', ['x', 'F'])(x, F)
        self._visualization = Collection(objects)
        self._update()
        return self._visualization
