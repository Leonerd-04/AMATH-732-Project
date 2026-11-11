from typing import Callable
import numpy as np

class Individual:
    # Initializes an individual with its position
    def __init__(self, x:float, y:float, infected: bool):
        self.x = x
        self.y = y
        self.infected = infected
        self.masked = False
        self.masked_x = None
        self.masked_y = None

    # Calculate one step of brownian motion.
    # v is the volatility and represents how far the individuals move in a given timestep
    def move_brownian(self, v: float):
        # Generates an x and y displacement
        dx = v * np.random.normal(0, 1)
        dy = v * np.random.normal(0, 1)
        self.x += dx
        self.y += dy

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

    # If is infected, randomly disinfects.
    def tick_infection(self, prob_disinfection: float):
        if not self.infected:
            return
        if np.random.uniform(0, 1) < prob_disinfection:
            self.infected = False


    # Check whether to put on/take off a mask
    def mask_check(self, masking_probability: float, mask_multiplier: float):
        if self.masked_x is None:
            return

        dist_sick_sq = (self.x - self.masked_x) ** 2 + (self.y - self.masked_y) ** 2
        is_in_sick_range = dist_sick_sq < 0.1

        current_masking_probability = masking_probability

        if is_in_sick_range:
            current_masking_probability = masking_probability * mask_multiplier

        if np.random.uniform(0, 1) < current_masking_probability:
            self.masked = True
        else:
            self.masked = False


    # Notifies an individual they've been infected
    def infect(self):
        self.infected = True
        self.masked_x = self.x
        self.masked_y = self.y