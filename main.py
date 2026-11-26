
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
rotational_speed = 4    # How quick the rotation is around the annulus
Rdispersion = 0.1       # How far apart the individuals start from each other
dt = 0.01
v = 0.005               # Volatility, describes the magnitude of the brownian motion

interaction_distance = 0.05     # How close to be to trigger an exchange of opinion
interaction_strength = 0.3      # How strong this opinion change is
echo_effect = 0.06              # When two people are similar opinion strengths, this term pushes their collective opinions up.

prop_initial_infection = 0.2    # (Deprecated) The proportion of individuals that are infected at t = 0
infection_probability = 0.3     # (Deprecated) Probability of being infected given that an individual is within infection distance
mask_reduction = 0.6            # (Deprecated) How much a mask reduces probability of infection

masking_probability = 0.01      # Deprecated
mask_multiplier = 10            # Deprecated

infection_halflife = 0.9999

# The probability of being cured at any given time step
# With it being modeled as an exponential distribution
# Deprecated
prob_disinfection = 1 - np.exp2(-dt/infection_halflife)


# Function for keeping individuals in annulus
# Defined in r and theta
R0 = 1
theta0 = 0
strength = 0.1
def u_annulus(r:float, theta: float)-> tuple[float, float]:
    ur = -strength * (1 + 99 * np.exp(-((theta - theta0)/0.4) ** 2)) * (r - R0)
    uth = rotational_speed      # Here the rotation is just lumped into our velocity field
    return ur, uth

# Function for keeping individulas in two orbiting circles.
R0 = 0.5
def u_2_annuli(x:float, y: float)-> tuple[float, float]:
    r_1 = np.sqrt((x - R0)** 2 + (y) ** 2)
    theta_1 = np.arctan2(y, x - R0)
    r_2 = np.sqrt((x + R0)** 2 + (y) ** 2)
    theta_2 = np.arctan2(y, x + R0)

    ur1 = -strength * (1 + 99 * np.exp(-((theta_1 - np.pi)/0.4) ** 2)) * (r_1 - R0) * np.exp( - r_1 ** 2)
    uth1 = rotational_speed

    ur2 = -strength * (1 + 99 * np.exp(-((theta_2 - 0)/0.4) ** 2)) * (r_2 - R0) * np.exp( - r_2 ** 2)
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
            "opinion": indiv.opinion,
            "masked": indiv.masked,
            "ever_opinionated": ever_infected[i],
            "x": indiv.x,
            "y": indiv.y
        })


def init_individuals_groups():
    for i in range(numparts):
        # Generate a random starting position using polar coordinates
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(R0 - Rdispersion, R0 + Rdispersion)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        if np.random.uniform(0, 1) < 0.5:
            x += R0
        else:
            x -= R0

        if np.abs(y/x) < 1/6 and x > 0.2:
            opinion = np.random.uniform(-1, 0)
        elif np.abs(y/x) < 1/6 and x < 0.2:
            opinion = np.random.uniform(0, 1)
        else:
            opinion = 0

        individuals.append(Individual(x, y, opinion))
        ever_infected.append(opinion != 0)


def init_individuals_uniform():
    for i in range(numparts):
        # Generate a random starting position using polar coordinates
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(R0 - Rdispersion, R0 + Rdispersion)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        if np.random.uniform(0, 1) < 0.5:
            x += R0
        else:
            x -= R0

        opinion = np.random.normal(0, 0.6)
        individuals.append(Individual(x, y, opinion))
        ever_infected.append(opinion != 0)


# As opposed to plots.py, renders an animation of the individuals as the code is running.
def display(fig, ax, t:float):
    ax.clear()

    # Grab the x and y coordinates
    x = np.array([p.x for p in individuals])
    y = np.array([p.y for p in individuals])

    infected = np.array([p.opinion for p in individuals])
    masked = np.array([p.masked for p in individuals])

    # Uninfected individuals in black
    # ax.plot(
    #   x[infected == 0],
    #    y[infected == 0],
    #    'k.',
    #    markersize=3
    #)

    # Infected individuals using a colour gradient
    ax.scatter(
        x,
        y,
        c=infected,
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
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    plt.pause(0.001)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def main():
    init_individuals_uniform()
    fig, ax = plt.subplots(figsize=(7, 7))
    sm = plt.cm.ScalarMappable(
        cmap='RdYlBu_r',
        norm=plt.Normalize(vmin=-1, vmax=1)
    )

    sm.set_array([])

    fig.colorbar(sm, ax=ax, label="Polarization Strength")
    t = 0

    # Open CSV files for data logging
    os.makedirs("results", exist_ok=True)

    with open("results/individual_opinions.csv", "w", newline="") as f, \
         open("results/new_opinions.csv", "w", newline="") as nf:

        fieldnames = [
            "time",
            "agent_num",
            "opinion",
            "masked",
            "ever_opinionated",
            "x",
            "y"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        new_fieldnames = [
            "time",
            "agent_num",
            "opinion_strength",
            "masked",
            "x",
            "y"
        ]

        new_writer = csv.DictWriter(nf, fieldnames=new_fieldnames)
        new_writer.writeheader()

        # Log initial agent states
        log_individuals(t, writer)

        # This loop handles all of the simulation stepping.
        for i in range(numsteps):
            display(fig, ax, t)
            t += dt

            # Get coordinates in a suitable format, then find the distance matrix
            # Such that dist[i, j] is the distance between the ith and jth individual
            x = [[p.x, p.y] for p in individuals]
            dist = sp.distance.squareform(sp.distance.pdist(x))

            new_infection_flags = np.zeros(numparts, dtype=bool)

            for i, indiv in enumerate(individuals):
                # Movement
                indiv.move(dt, v, u_2_annuli, is_polar=False)

                # # Check for infection
                # indiv.tick_infection(prob_disinfection)

                # Check whether to wear a mask
                # indiv.mask_check(masking_probability, mask_multiplier)

                # # Skips checking for infection if the given individual is infected already
                # if indiv.infected:
                #     continue

                # Check whether to become infected
                # An individual becomes infected when they are within the infection distance of another
                for j in range(numparts):
                    if i == j:
                        continue

                    if individuals[j].opinion == 0 or dist[i, j] > interaction_distance:
                        continue

                    p_infection = np.exp(-1 * np.abs(individuals[j].opinion - individuals[i].opinion))

                    # if indiv.masked:
                    #     # p_infection *= mask_reduction
                    #     pass

                    if p_infection < np.random.uniform(0, 1):
                        continue

                    infection_level =  ((1 + echo_effect) * individuals[j].opinion - individuals[i].opinion) * np.random.uniform(0, interaction_strength)
                    indiv.change_opinion(infection_level)

                    if not ever_infected[i]:
                        ever_infected[i] = True
                        new_infection_flags[i] = True

                        new_writer.writerow({
                            "time": t,
                            "agent_num": i,
                            "opinion_strength": infection_level,
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
