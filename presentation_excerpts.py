from math import copysign
from prettytable.colortable import ColorTable, Themes
from typing import NamedTuple
import matplotlib.pyplot as plt

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
    c, k, f = s
    print(
        f"t={k.t:.1f}, f_d={f.drag:.1f}, f_s={f.spring:.1f}, f_net={f.net:.1f}, a={k.a:.1f}, v={k.v:.1f}, x={k.x:.1f}"
    )


def init(delta_t, x, mass, drag_coeff, area, k, coil_length, v=0.0) -> SimState:
    """
    create initial row from global constants and initial values kinematic
    values
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
    c, k, _ = prev  # unpack SimState

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
    dir_v = copysign(1, k.v)
    direction_drag = dir_v * -1
    # compute positive drag force and apply `direction_drag` to ensure it's
    # opposite
    return direction_drag * c.drag_coeff * (k.v**2) * c.area


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
        print("t = ", k.t, " disp = ", displacement, "spring_force = ", F_s)
        return F_s

        # use this alternate condition to disable spring force when ball passes
        # (moving upwards) the lower equilibrium point (with ball mass). This
        # is what Marcus suggested we use, but it causes the model to lose
        # energy (work done by the spring over the distance between the two
        # equilibrium  points is lost)

        # with the lossy model, the graphs get weird af if the ball is dropped
        # too close to the spring. we have no idea why

    elif k.v > 0 and k.x <= equilibrium_w_mass(c):
        # elif k.v >= 0 and k.x <= c.coil_length:
        # moving up and in contact with spring, release force at lower
        # equilibrium point
        print("t = ", k.t, " disp = ", displacement, "spring_force = ", F_s)
        return F_s
    else:
        print("t = ", k.t, " disp = ", displacement, "spring_force = ", 0.0)
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
ts = []  # timesteps
xs = []  # positions
vs = []  # vel
accs = []
spring_forces = []
f_nets = []
drags = []

prev = init(
    delta_t=0.002,
    x=19.0, 
    mass=1.0,
    drag_coeff=0.0,
    area=0.0,
    k=10,
    coil_length=17.0,
    v=0.0,
)
num_steps = 10000
for i in range(0, num_steps):
    next = step_sim(prev)
    prev = next


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

# spring force
make_plot(
    "Drag Force vs Time",
    "drag_force.png",
    x_label="t (s)",
    y_label="F (N)",
    x_data=ts,
    y_data=drags,
)

# # spring force
# make_plot(
#     "NetForce vs Time",
#     "net_force.png",
#     x_label="t (s)",
#     y_label="F (N)",
#     x_data=ts,
#     y_data=f_nets,
# )
#
plt.show()

# generate human readable table

table = ColorTable(theme=Themes.OCEAN)

# add columns
table.add_column("t (s)", ts)
table.add_column("v (m/s)", vs)
table.add_column("a (m/s^2)", accs)
table.add_column("x (m)", xs)
table.add_column("f_net (N)", f_nets)
table.add_column("F_S (N)", spring_forces)
table.add_column("Drag (N)", drags)

table.align = "r"
table.float_format = ".1"

# table -> html
table.format = True
table.border = True

html = table.get_html_string()
with open("table.html", "w") as f:
    f.write(html)
print(table)
