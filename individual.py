from typing import Callable
import numpy as np

class Individual:
    # Initializes an individual with its position
    def __init__(self, x:float, y:float, infected:float):
        self.x = x
        self.y = y
        self.opinion = float(infected)
        self.masked = False     # Deprecated, but hard to remove

    # Calculate one step of brownian motion.
    # v is the volatility and represents how far the individuals move in a given timestep
    def move_brownian(self, v: float):
        # Generates an x and y displacement
        dx = v * np.random.normal(0, 1)
        dy = v * np.random.normal(0, 1)
        self.x += dx
        self.y += dy

    def move_opinionated(self, sigma: float, vol: float, dt: float):
        # If close to the center, move to the left or right depending on opinion.
        # Helps create "echo chamber" effect, you tend to hang with those you associate with.
        if np.abs(self.x) < sigma * 1:
            u = vol * np.clip(self.opinion, -1, 1) * np.exp(-(self.x / sigma) ** 2)
            self.x += u * dt



    # calculates one movement step
    # v is the volatility and represents how far the individuals move in a given timestep
    # dt is the timestep length
    # u is a velocity field u(x, y) = (ux, uy) which the individuals follow
    # unless u is defined using r and theta, then we have u(r, theta) = ur, utheta
    # is_polar specifies whether we are using x and y or r and theta
    def move(self, dt: float, v: float, u: Callable[[float, float], tuple[float, float]], is_polar: bool = False):
        self.move_brownian(v)
        if is_polar:
            # Convert to polar
            r = np.sqrt(self.x ** 2 + self.y ** 2)
            theta = np.arctan2(self.y, self.x)

            # Apply the velocity in polar coords
            ur, ut = u(r, theta)
            r += ur * dt
            theta += ut * dt

            # Convert back to cartesian
            self.x = r * np.cos(theta)
            self.y = r * np.sin(theta)
        else:
            ux, uy = u(self.x, self.y)  # Use the provided field to redirect movement
            self.x += ux * dt
            self.y += uy * dt
        pass
        self.move_opinionated(0.01, 0.8, dt)

    # Notifies an individual they've been infected
    def change_opinion(self, delta_op: float = 1.0):
        self.opinion += delta_op
        self.opinion = np.clip(self.opinion, -1, 1)