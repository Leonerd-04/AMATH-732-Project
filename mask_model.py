
import numpy as np


class MaskModel:

    def __init__(self, config):

        self.mask_prob = config.mask_prob
        self.mask_off_prob = config.mask_off_prob
        self.mask_mult = config.mask_mult
        self.learning_dist = config.learning_dist

    def update(self, particles):

        n = len(particles)

        # Calculate probability of adopting a mask
        probabilities = np.full(n, self.mask_prob)

        for i, particle in enumerate(particles):

            if particle.x_sick is not None:

                dist = np.sqrt(
                    (particle.x - particle.x_sick)**2
                    +
                    (particle.y - particle.y_sick)**2
                )

                if dist < self.learning_dist:

                    probabilities[i] *= self.mask_mult

        # Random decisions to put masks on
        put_mask = np.random.rand(n) < probabilities

        # Random decisions to take masks off
        remove_mask = np.random.rand(n) < self.mask_off_prob

        # Update mask states
        for i, particle in enumerate(particles):

            if not particle.mask_on:

                if put_mask[i]:
                    particle.mask_on = True

            elif remove_mask[i]:

                particle.mask_on = False
