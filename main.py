from individual import Individual
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp

# Important parameters
numparts = 1000
numsteps = 1000
mysig = 0.1
rotational_speed = 1    # How quick the rotation is around the annulus
Rdispersion = 0.1       # How far apart the individuals start from each other
dt = 0.01
v = 0.005               # Volatility

prop_initial_infection = 0.05   # The proportion of individuals that are infected at t = 0
infection_distance = 0.01       # How close to be to trigger infection
infection_probability = 0.1     # Probability of being infected given that an individual is within infection distance
mask_reduction = 0.6            # How much a mask reduces probability of infection

masking_probability = 0.01
mask_multiplier = 10

infection_halflife = 0.01

# The probability of being cured at any given time step
# With it being modeled as an exponential distribution
prob_disinfection = 1 - np.exp2(dt/infection_halflife)


# Function for keeping individuals in annulus
# Defined in r and theta
R0 = 1
theta0 = 0
strength = 0.1
def u_annulus(r:float, theta: float)-> tuple[float, float]:
    ur = -strength * (1 + 99 * np.exp(-((theta - theta0)/0.4) ** 2)) * (r - R0)
    uth = rotational_speed      # Here the rotation is just lumped into our velocity field
    return ur, uth

individuals = []

def init_individuals():
    for i in range(numparts):
        # Generate a random starting position using polar coordinates
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(R0 - Rdispersion, R0 + Rdispersion)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # First 5% of individuals will be infected
        infected = i/numparts < 0.05

        individuals.append(Individual(x, y, infected))

def display(fig, ax, t:float):
    ax.clear()

    # Grab the x and y coordinates for the individuals
    x = np.array([p.x for p in individuals])
    y = np.array([p.y for p in individuals])

    infected = np.array([
        p.infected for p in individuals
    ])

    masked = np.array([
        p.masked for p in individuals
    ])

    # Plot as a scatter plot
    ax.plot(x, y, 'k.', markersize=3)

    # Infected individuals
    ax.plot(
        x[infected],
        y[infected],
        'ro',
        markersize=5
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
    init_individuals()
    fig, ax = plt.subplots(figsize=(7, 7))
    t = 0
    # This loop handles all of the simulation stepping.
    for i in range(numsteps):
        display(fig, ax, t)
        t += dt

        # Get coordinates in a suitable format, then find the distance matrix
        # Such that dist[i, j] is the distance between the ith and jth individual
        x = [[p.x, p.y] for p in individuals]
        dist = sp.distance.squareform(sp.distance.pdist(x))

        for i, indiv in enumerate(individuals):
            # Movement
            indiv.move(dt, v, u_annulus, is_polar=True)

            # Check for infection
            indiv.tick_infection(prob_disinfection)

            # Check whether to wear a mask
            indiv.mask_check(masking_probability, mask_multiplier)

            # Skips checking for infection if the given individual is infected already
            if indiv.infected:
                continue

            # Check whether to become infected
            # An individual becomes infected when they are within the infection distance of another
            for j in range(numparts):
                if individuals[j].infected and dist[i, j] < infection_distance:
                    p_infection = infection_probability

                    if indiv.masked:
                        p_infection *= mask_reduction

                    if np.random.uniform(0, 1) < p_infection:
                        indiv.infect()




    # Plot non-interactively
    plt.ioff()
    plt.show()
if __name__ == "__main__":
    main()