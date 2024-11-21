from math import copysign
import matplotlib.pyplot as plt

# GLOBALS
g = -9.8  # m/s^2
MAX_STEPS = 100


# TYPES
class State:
    """
    Represents the state that's threaded through 'step' (in essence the data in
    our state monad, `step :: StateT IO Row`)
    """

    @staticmethod
    def initial(
        initial_height,
        mass,
        area,
        k,
        coil_length,
        delta_t=0.1,
        drag_coeff=0.0,
        t=0,
        v=0.0,
        a=g,
    ):
        """
        Provide a few sensible defaults when instantiating initial state.
        """
        return State(
            t, v, a, initial_height, mass, drag_coeff, area, delta_t, k, coil_length
        )

    def __init__(
        self, t, v, a, x, mass, drag_coeff, area, delta_t, k, coil_length
    ) -> None:
        """
        Ye 'ol constructor

        Should not need to use directly, use `State.initial(...)` to get first `State`.
        """

        # non-column values have (in some cases sensible) defaults
        self.mass = mass  # kg
        self.drag_coeff = drag_coeff
        self.area = area  # m^2
        self.delta_t = delta_t
        self.k = k  # spring constant
        self.coil_length = coil_length  # length of spring coil

        # column values
        self.t = t
        self.v = v
        self.f_d = self.drag_force()  # this is computed from drag_coeff
        self.f_g = g * self.mass
        self.a = a
        self.x = x
        self.f_s = self.spring_force()
        self.f_net = self.f_d + self.f_g

    def __repr__(self) -> str:
        return f"State(t={self.t:.1f}, f_g={self.f_g:.1f}, f_d={self.f_d:.1f}, f_s={self.f_s:.1f}, f_net={self.f_net:.1f}, a={self.a:.1f}, v={self.v:.1f}, x={self.x:.1f})"

    def copy(self):
        return State(
            self.t,
            self.v,
            self.a,
            self.x,
            self.mass,
            self.drag_coeff,
            self.area,
            self.delta_t,
            self.k,
            self.coil_length,
        )

    def drag_force(self) -> float:
        """
        Computes drag force as function of velocity, `drag_coeff`, `area`, and
        `mass`

        The direction of the force is opposite that of `vel`.

        let drag_coeff be the drag force coefficient and `area` be the surace area
        of our object of mass `mass`; then
        - `F_drag = drag_coeff * v^2 * area`
        """
        # invert velocity direction
        direction = copysign(1, self.v) * -1
        return direction * self.drag_coeff * (self.v**2) * self.area

    def net_force(self) -> float:
        return self.f_d + self.f_g + self.spring_force()

    def equilibrium_wo_mass(self) -> float:
        """
        Calculate equilibrium height of spring without mass

        Since the spring is massless, it won't compress under it's own weight,
        so the equilibrium point is simply `self.coil_length`
        """
        return self.coil_length

    def equilibrium_w_mass(self) -> float:
        """
        Calculate equilibrium height of spring with mass

        `coil_length - displacement` for a diplacement delta_x such that `F_g = F_s`

        Since `F_g = F_s` => `k * delta_x = m * g` => `delta_x = mg/k`.

        Then the equilibrium point is
        `coil_length - delta_x = coil_length - (mg/k)` meters off the ground.
        """
        return self.coil_length - (self.mass * (-g)) / self.k

    def spring_force(self) -> float:
        """
        Calculate spring force.

        Let `d` be the spring displacement.
        If `v < 0` (mass is moving downwards) and `x <= coil_length`,
        then apply spring force: `F_s = kd` where `d = coil_length - x`.
        """

        displacement = self.coil_length - self.x
        # spring force is always positive
        F_s = self.k * displacement

        if self.v <= 0 and self.x <= self.coil_length:
            # moving down and in contact with spring
            return F_s
        elif self.v > 0 and self.x <= self.equilibrium_w_mass():
            # moving up and in contact with spring, release force at lower
            # equilibrium point
            return F_s
        else:
            return 0.0

    # For now, consider only falling ball
    def step(self):  # -> State
        prev = self
        curr = prev.copy()
        # update fields

        curr.t = prev.t + self.delta_t

        # NOTE: order matters here

        # 1. update forces, sum to get net_force
        # - F_g = mass * g
        # - F_drag = drag_force(v)
        # TODO: spring force, damping
        curr.f_d = prev.drag_force()  # NB. coffee filter model used prev velocity
        curr.f_s = prev.spring_force()

        # 2. net_force -> accel
        # - F_net = F_drag - g
        curr.f_net = curr.net_force()  # use updated drag force

        # 3. accel -> vel -> disp
        curr.a = curr.f_net / self.mass  # accel is F_net/mass
        curr.v = prev.v + curr.a * self.delta_t  # delta_v = v_0 + at
        curr.x = prev.x + curr.v * self.delta_t  # delta_x = x_0 + vt

        return curr


# initialize lists to collect data for plotting

# timesteps
ts = []
# positions
xs = []

# vel
vs = []
accs = []



prev = State.initial(
        initial_height=15.0, mass=2.0, area=0.5, delta_t=0.1, k=15, coil_length=5, drag_coeff = 0.1
    )
print(prev.equilibrium_wo_mass())
print(prev.equilibrium_w_mass())
num_steps =  60
print(prev)
for i in range(0, num_steps):
    prev = prev.step()
    #print(prev)

    # collect data for plotting
    ts.append(prev.t)
    xs.append(prev.x)
    vs.append(prev.v)
    accs.append(prev.a)
    

# generate plots
fig, ax = plt.subplots()
ax.set_title("Position vs time")
ax.set_xlabel("t (s)")
ax.set_ylabel("x (m)")
ax.plot(ts, xs)
# if in jupyter, use:
#plt.show()
plt.savefig('xs.png')

# vel
fig, ax = plt.subplots()
ax.set_title("Velocity vs time")
ax.set_xlabel("t (s)")
ax.set_ylabel("v (m/s)")
ax.plot(ts, vs)
#plt.show()
plt.savefig('vs.png')

# accel
fig, ax = plt.subplots()
ax.set_title("Acceleration vs time")
ax.set_xlabel("t (s)")
ax.set_ylabel("a (m/s^2)")
ax.plot(ts, accs)
# plt.show()
plt.savefig('as.png')
