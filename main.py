from individual import Individual
import numpy as np
import matplotlib.pyplot as plt

# Important parameters
numparts = 1000
numsteps = 1000
mysig = 0.1
rotational_speed = 1    # How quick the rotation is around the annulus
Rdispersion = 0.1       # How far apart the individuals start from each other
dt = 0.01
v = 0.005

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
        individuals.append(Individual(x, y))

def display(fig, ax, t:float):
    ax.clear()

    # Grab the x and y coordinates for the individuals
    x = np.array([p.x for p in individuals])
    y = np.array([p.y for p in individuals])

    # Plot as a scatter plot
    ax.plot(x, y, 'k.', markersize=3)

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
    for i in range(numsteps):
        display(fig, ax, t)
        t += dt
        for indiv in individuals:
            indiv.move(dt, v, u_annulus, is_polar=True)

    plt.ioff()
    plt.show()
if __name__ == "__main__":
    main()