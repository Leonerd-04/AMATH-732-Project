
import numpy as np


class InfectionModel:

    def __init__(self, config):

        self.mindist0 = config.mindist0
        self.prob_inf0 = config.prob_inf0
        self.time_inf = config.time_inf
        self.mask_red = config.mask_red

    def update_infection_timers(self, particles):

        for particle in particles:
            particle.update_infection_timer()

    def calculate_distances(self, particles):

        positions = np.array([
            [p.x, p.y] for p in particles
        ])

        # Pairwise displacement vectors
        diff = positions[:, None, :] - positions[None, :, :]

        # Euclidean distances
        distances = np.sqrt(
            np.sum(diff**2, axis=2)
        )

        # Exclude self-distances
        np.fill_diagonal(distances, np.inf)

        return distances

    def update(self, particles):

        distances = self.calculate_distances(particles)

        # Minimum distance to another particle
        min_dist = np.min(distances, axis=1)

        # Calculate infection probabilities
        probabilities = np.array([
            self.prob_inf0 * (
                self.mask_red if p.mask_on else 1.0
            )
            for p in particles
        ])

        # Current infection states
        infected = np.array([
            p.infected for p in particles
        ])

        # Determine new infections simultaneously
        random_values = np.random.rand(len(particles))

        new_infections = (
            (min_dist < self.mindist0)
            & (random_values < probabilities)
            & (~infected)
        )

        # Apply new infections
        for i, particle in enumerate(particles):

            if new_infections[i]:
                particle.infect(self.time_inf)

