delta_t = 0.1  # s
mass = 2.0  # kg
initial_height = 20  # m
initial_velocity = 0  # m/s
# let c_d be the drag force coefficient and A be the surace area of our object
# of mass `mass`; then
# F_drag = c_d * v^2 * A
drag_force = 0

MAX_STEPS = 100


def step_time(t):
    return t + delta_t


def net_force_to_accel(net_force, mass):
    return net_force / mass


def delta_vel(accel):
    return accel * delta_t


# For now, consider only falling ball
def step(t):
    new_t = step_time(t)
    # 1. update forces, sum to get net_force

    # 2. net_force -> accel
    # 3. accel -> vel -> disp
    # acceleration is F_net/mass

    print("stepped from", t, "to", new_t)


if __name__ == "__main__":
    step(0)
