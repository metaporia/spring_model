from contextlib import ContextDecorator
from math import copysign
import matplotlib.pyplot as plt

from collections import namedtuple
from typing import NamedTuple
from dataclasses import dataclass

# GLOBALS
g = -9.8  # m/s^2
MAX_STEPS = 100


# DATATYPES


class Kin(NamedTuple):
    """
    values that need to be tracked throughout
    """

    t: float
    v: float
    a: float
    x: float


# initial conditions (per run)
class InitialConditions(NamedTuple):
    delta_t: float
    mass: float
    drag_coeff: float
    area: float
    k: float
    coil_length: float


class Forces(NamedTuple):
    grav: float
    drag: float
    spring: float
    net: float

    @staticmethod
    def new(grav, drag, spring):
        """
        Creat new `Forces` object. Compute net force.
        """
        return Forces(grav, drag, spring, net=grav + drag + spring)


# (conditions, kin, forces)
class SimState(NamedTuple):
    c: InitialConditions
    k: Kin
    f: Forces

def debug_state(s: SimState):
    c,k,f = s
    print(f"t={k.t:.1f}, f_d={f.drag:.1f}, f_s={f.spring:.1f}, f_net={f.net:.1f}, a={k.a:.1f}, v={k.v:.1f}, x={k.x:.1f}")


def init(delta_t, x, mass, drag_coeff, area, k, coil_length, v=0.0) -> SimState:
    """
    create
    """
    c = InitialConditions(
        delta_t,
        mass,
        drag_coeff,
        area,
        k,
        coil_length,
    )
    k = Kin(t=0, v=v, a=g, x=x)
    f = Forces.new(grav=mass * g, drag=drag_force(c, k), spring=spring_force(c, k))
    return SimState(c, k, f)


def step_sim(prev: SimState) -> SimState:
    """
    Perform a single step of the simulation, bumping time by `delta_t` updating
    kinematic values and forces.

    Return next state.
    """

    c, k, _ = prev  # unpack SimState

    # 'A <- B' means B depends on the value of A
    # the dependency graph is (simplified):
    # (f_g, f_d, f_s) <- f_net <- a <- v <- x
    # where x <- f_d, (v, x) <- f_s
    #
    # So we'll calcualte spring and drag forces from the previous state,
    # and then update Kin (a, v, x) from the resultant net force.
    #
    # It's a cyclical dependency graph, so there will be a delay in the
    # updating of some values no matter how the graph is ordered. Decreasing
    # `delta_t` will reduce the inaccuracy introduced by the cycle.

    drag = drag_force(prev.c, prev.k)
    spring = spring_force(prev.c, prev.k)
    # `Forces.new` calculates net force
    new_forces = Forces.new(prev.f.grav, drag, spring)

    # update kinematic values from forces
    t = k.t + c.delta_t  # bump time
    a = new_forces.net / c.mass
    v = prev.k.v + a * c.delta_t
    x = prev.k.x + v * c.delta_t

    new_k = Kin(t=t, a=a, v=v, x=x)

    # create next state
    next = SimState(c, new_k, new_forces)
    return next


def drag_force(c: InitialConditions, k: Kin) -> float:
    """
    Computes drag force as function of velocity, `drag_coeff`, `area`, and
    `mass`

    The direction of the force is opposite that of `vel`.

    let drag_coeff be the drag force coefficient and `area` be the surace area
    of our object of mass `mass`; then
    - `F_drag = drag_coeff * v^2 * area`
    """
    # invert velocity direction
    direction = copysign(1, k.v) * -1
    return direction * c.drag_coeff * (k.v**2) * c.area


def spring_force(c: InitialConditions, k: Kin) -> float:
    """
    Calculate spring force.

    Let `d` be the spring displacement.
    If `v < 0` (mass is moving downwards) and `x <= coil_length`,
    then apply spring force: `F_s = kd` where `d = coil_length - x`.
    """

    displacement = c.coil_length - k.x
    # spring force is always positive
    F_s = c.k * displacement

    if k.v <= 0 and k.x <= c.coil_length:
        # moving down and in contact with spring
        return F_s
    elif k.v > 0 and k.x <= equilibrium_w_mass(c):
        # moving up and in contact with spring, release force at lower
        # equilibrium point
        return F_s
    else:
        return 0.0


def equilibrium_w_mass(c: InitialConditions) -> float:
    """
    Calculate equilibrium height of spring with mass

    This represents the release point when rebounding (i.e., moving upwards).

    `coil_length - displacement` for a diplacement delta_x such that `F_g = F_s`

    Since `F_g = F_s` => `k * delta_x = m * g` => `delta_x = mg/k`.

    Then the equilibrium point is
    `coil_length - delta_x = coil_length - (mg/k)` meters off the ground.
    """
    return c.coil_length - (c.mass * (-g)) / c.k


# Visualization
# initialize lists to collect data for plotting

# timesteps
ts = []
# positions
xs = []

# vel
vs = []
accs = []

spring_forces = []


# change starting conditions here
prev = init(
    delta_t=0.1,
    x=15.0,
    mass=1.0,
    drag_coeff=0.01,
    area=0.5,
    k=10,
    coil_length=8.0,
    v=0.0,
)
num_steps = 100 

# coil_length is upper equilibrium point
print(prev.c.coil_length)
print(equilibrium_w_mass(prev.c))  # lower equilibrium point
debug_state(prev)
for i in range(0, num_steps):
    next = step_sim(prev)
    debug_state(next)

    # collect data for plotting
    # to add a graph, just initialize a list above (under comment "Visualization")
    # and append the value you want to track here.
    ts.append(next.k.t) # time
    xs.append(next.k.x) # postiion
    vs.append(next.k.v) # velocity
    accs.append(next.k.a) # acceleration
    spring_forces.append(next.f.spring)
    prev = next


# # generate plots
# fig, ax = plt.subplots()
# ax.set_title("Position vs time")
# ax.set_xlabel("t (s)")
# ax.set_ylabel("x (m)")
# ax.plot(ts, xs)
# # if in jupyter, use:
# #plt.show()
# plt.savefig('xs.png')
# # vel
# fig, ax = plt.subplots()
# ax.set_title("Velocity vs time")
# ax.set_xlabel("t (s)")
# ax.set_ylabel("v (m/s)")
# ax.plot(ts, vs)
# #plt.show()
# plt.savefig('vs.png')
# # accel
# fig, ax = plt.subplots()
# ax.set_title("Acceleration vs time")
# ax.set_xlabel("t (s)")
# ax.set_ylabel("a (m/s^2)")
# ax.plot(ts, accs)
# # plt.show()
# plt.savefig('as.png')
#




def make_plot(title: str, filename: str, x_label: str, y_label: str, x_data, y_data):
    """
    Generate a plot.
    Collect y_data in the for loop above that steps through the sim
    """
    _, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(x_data, y_data)

    if filename != "":
        plt.savefig(filename)


# position
make_plot(
    "Position vs Time",
    "position.png",
    x_label="t (s)",
    y_label="x (m)",
    x_data=ts,
    y_data=xs,
)

# if in jupyter, use:
# plt.show()

# position
make_plot(
    "Velocity vs Time",
    "velocity.png",
    x_label="t (s)",
    y_label="v (m/s)",
    x_data=ts,
    y_data=vs,
)


# acceleration
make_plot(
    "Acceleration vs Time",
    "acceleration.png",
    x_label="t (s)",
    y_label="a (m/s^2)",
    x_data=ts,
    y_data=accs,
)

# spring force
make_plot(
    "Spring Force vs Time",
    "spring_force.png",
    x_label="t (s)",
    y_label="F (N)",
    x_data=ts,
    y_data=spring_forces,
)

plt.show()
