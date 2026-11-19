from individual import Individual
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
import csv
import os

# Important parameters
numparts = 1000
numsteps = 1000
mysig = 0.1
rotational_speed = 3
Rdispersion = 0.1
dt = 0.01
v = 0.05

# Ring geometry. Each ring passes through the origin (its radius equals its
# center offset), so both traps sit at the origin regardless of size.
# Larger R_S means a bigger circle AND a farther-off-center; the infection
# wave therefore travels farther and arrives later.
R_right = 1/2     # right ring (opinion +1)
R_left = 1/2      # left ring (opinion -1)

# Initial opinion in an angular segment of each circle (per-circle control)
initial_opinion_right = 1.0
initial_opinion_left = -1.0
initial_opinion_theta_center_right = 0.0
initial_opinion_theta_center_left = 0.0
initial_opinion_theta_width_right = 0.1
initial_opinion_theta_width_left = 0.5

infection_distance = 0.05
infection_probability = 0.1
mask_reduction = 0.

masking_probability = 0.0
mask_multiplier = 1

infection_halflife = 1

prob_disinfection = 1 - np.exp2(-dt/infection_halflife)

# Function for keeping individuals in annulus
R0 = 1
theta0 = 0
strength = 0.1

def u_annulus(r: float, theta: float) -> tuple[float, float]:
    ur = -strength * (1 + 99 * np.exp(-((theta - theta0)/0.4) ** 2)) * (r - R0)
    uth = rotational_speed
    return ur, uth


# Function for keeping individuals in two orbiting circles, each with its
# own radius (and hence its own center offset, so both pass through origin).
def u_2_annuli(x: float, y: float) -> tuple[float, float]:
    r_1 = np.sqrt((x - R_right)**2 + y**2)
    theta_1 = np.arctan2(y, x - R_right)

    r_2 = np.sqrt((x + R_left)**2 + y**2)
    theta_2 = np.arctan2(y, x + R_left)

    ur1 = -strength * (1 + 99 * np.exp(-((theta_1 - np.pi)/0.4) ** 2)) * (r_1 - R_right) * np.exp(-r_1**2)
    uth1 = rotational_speed

    ur2 = -strength * (1 + 99 * np.exp(-((theta_2 - 0)/0.4) ** 2)) * (r_2 - R_left) * np.exp(-r_2**2)
    uth2 = -rotational_speed

    u1 = ur1 * np.cos(theta_1) - uth1 * np.sin(theta_1)
    v1 = ur1 * np.sin(theta_1) + uth1 * np.cos(theta_1)

    u2 = ur2 * np.cos(theta_2) - uth2 * np.sin(theta_2)
    v2 = ur2 * np.sin(theta_2) + uth2 * np.cos(theta_2)

    if x < 0:
        return u2, v2
    else:
        return u1, v1


individuals = []
ever_infected = []



# Logging
def log_individuals(t, writer):

    for i, indiv in enumerate(individuals):

        writer.writerow({
            "time": t,
            "agent_num": i,
            "infection": indiv.infected,
            "masked": indiv.masked,
            "ever_infected": ever_infected[i],
            "x": indiv.x,
            "y": indiv.y
        })


def init_individuals():

    for i in range(numparts):

        if np.random.uniform(0, 1) < 0.5:
            side = "right"
            ring_radius = R_right
            x_offset = R_right
        else:
            side = "left"
            ring_radius = R_left
            x_offset = -R_left

        # Generate a random starting position using polar coordinates
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(ring_radius - Rdispersion, ring_radius + Rdispersion)

        x = x_offset + r * np.cos(theta)
        y = r * np.sin(theta)

        # Set opinion for agents whose angular position falls within the
        # specified segment of each circle. `theta` is the polar angle
        # around the respective circle's center.
        if side == "right":
            theta_c = initial_opinion_theta_center_right
            width = initial_opinion_theta_width_right
            opinion = initial_opinion_right
        else:
            theta_c = initial_opinion_theta_center_left
            width = initial_opinion_theta_width_left
            opinion = initial_opinion_left

        infected = 0.0
        delta = (theta - theta_c + np.pi) % (2*np.pi) - np.pi
        if abs(delta) <= width / 2:
            infected = opinion

        individuals.append(Individual(x, y, infected))
        ever_infected.append(infected != 0)


def display(fig, ax, t: float):

    ax.clear()

    # Grab the x and y coordinates
    x = np.array([p.x for p in individuals])
    y = np.array([p.y for p in individuals])

    infected = np.array([p.infected for p in individuals])
    masked = np.array([p.masked for p in individuals])

    # Uninfected individuals in black
    ax.plot(
        x[infected == 0],
        y[infected == 0],
        'k.',
        markersize=3
    )

    # Infected individuals using a colour gradient
    ax.scatter(
        x[infected != 0],
        y[infected != 0],
        c=infected[infected != 0],
        cmap='RdYlBu_r',
        vmin=-1,
        vmax=1,
        s=25
    )

    # Masked individuals
    ax.plot(
        x[masked],
        y[masked],
        'g*',
        markersize=6
    )

    ax.grid(True)
    ax.set_title(f"Time = {t:.3f}")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect('equal')

    plt.pause(0.001)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def main():

    init_individuals()

    fig, ax = plt.subplots(figsize=(7, 7))

    sm = plt.cm.ScalarMappable(
        cmap='RdYlBu_r',
        norm=plt.Normalize(vmin=-1, vmax=1)
    )

    sm.set_array([])

    fig.colorbar(sm, ax=ax, label="Infection strength")

    t = 0

    # Open CSV file for agent-level logging
    os.makedirs("results", exist_ok=True)

    with open("results/individual_infections.csv", "w", newline="") as f, \
         open("results/new_infections.csv", "w", newline="") as nf:

        fieldnames = [
            "time",
            "agent_num",
            "infection",
            "masked",
            "ever_infected",
            "x",
            "y"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        new_fieldnames = [
            "time",
            "agent_num",
            "infection_strength",
            "masked",
            "x",
            "y"
        ]

        new_writer = csv.DictWriter(nf, fieldnames=new_fieldnames)
        new_writer.writeheader()

        # Log initial agent states
        log_individuals(t, writer)

        # Simulation loop (one timestep = one phase sweep)
        for step in range(numsteps):

            display(fig, ax, t)
            t += dt

            # Snapshot opinions at start of step (pre-decay, pre-spread).
            s_old = np.array([p.infected for p in individuals])

            # Phase 1: move, decay, mask — applied to all agents.
            for p in individuals:
                p.move(dt, v, u_2_annuli, is_polar=False)
                p.tick_infection(prob_disinfection)
                p.mask_check(masking_probability, mask_multiplier)

            # Phase 2: distances after movement (fresh positions).
            pos = [[p.x, p.y] for p in individuals]
            dist = sp.distance.squareform(sp.distance.pdist(pos))

            new_infection_flags = np.zeros(numparts, dtype=bool)

            # Phase 3: spread using pre-decay opinions.
            for i, indiv in enumerate(individuals):

                # Skip checking for infection if already infected
                if indiv.infected:
                    continue

                # Check whether to become infected
                for j in range(numparts):

                    if i == j:
                        continue

                    if s_old[j] != 0 and dist[i, j] < infection_distance:

                        p_infection = infection_probability * abs(s_old[j])

                        if indiv.masked:
                            p_infection *= mask_reduction

                        if np.random.uniform(0, 1) < p_infection:

                            infection_level = np.sign(s_old[j]) * np.random.uniform(0, 1)
                            indiv.infect(infection_level)

                            if not ever_infected[i]:
                                ever_infected[i] = True
                                new_infection_flags[i] = True

                                new_writer.writerow({
                                    "time": t,
                                    "agent_num": i,
                                    "infection_strength": infection_level,
                                    "masked": indiv.masked,
                                    "x": indiv.x,
                                    "y": indiv.y
                                })

                            break

            # Log all agents after the timestep has completed
            log_individuals(t, writer)

    # Plot non-interactively
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()