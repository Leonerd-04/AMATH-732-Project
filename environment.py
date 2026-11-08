
import numpy as np


class Environment:

    def __init__(self, config):

        self.R0 = config.R0
        self.theta0 = config.theta0
        self.pull0 = config.pull0
        self.rotspd = config.rotspd

    def radial_pull(self, theta):

        return -self.pull0 * (
            1 + 99 * np.exp(
                -((theta - self.theta0) / 0.4)**2
            )
        )

    def brownian_motion(self, particles, dt, sigma):

        noise_scale = np.sqrt(dt) * sigma

        for particle in particles:

            particle.x += noise_scale * np.random.randn()
            particle.y += noise_scale * np.random.randn()

    def update_position(self, particle, dt):

        # Convert to polar coordinates
        r = np.sqrt(particle.x**2 + particle.y**2)

        theta = np.arctan2(
            particle.y,
            particle.x
        )

        # Radial restoring force
        r += dt * self.radial_pull(theta) * (r - self.R0)

        # Angular rotation
        theta += self.rotspd * dt

        # Convert back to Cartesian coordinates
        particle.x = r * np.cos(theta)
        particle.y = r * np.sin(theta)
