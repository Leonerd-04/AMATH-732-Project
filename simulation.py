
import numpy as np

from particle import Particle
from environment import Environment
from infection_model import InfectionModel
from mask_model import MaskModel
from visualizer import Visualizer


class Simulation:

    def __init__(self, config):

        self.config = config

        self.t = 0.0

        # Initialize random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Initialize particles
        self.particles = self.initialize_particles()

        # Initialize infected population
        self.initialize_infections()

        # Initialize model components
        self.environment = Environment(config)

        self.infection_model = InfectionModel(config)

        self.mask_model = MaskModel(config)

        self.visualizer = Visualizer()

    # --------------------------------------------------------
    # Initialize particle positions
    # --------------------------------------------------------

    def initialize_particles(self):

        n = self.config.numparts

        r = 0.8 + 0.4 * np.random.rand(n)

        theta = (
            -np.pi
            + 2 * np.pi * np.random.rand(n)
        )

        particles = []

        for i in range(n):

            x = r[i] * np.cos(theta[i])
            y = r[i] * np.sin(theta[i])

            particles.append(
                Particle(x, y)
            )

        return particles

    # --------------------------------------------------------
    # Initialize infections
    # --------------------------------------------------------

    def initialize_infections(self):

        numinf = self.config.numinf
        n = self.config.numparts

        infected_indices = np.random.choice(
            n,
            numinf,
            replace=True
        )

        for i in infected_indices:

            self.particles[i].infect(
                self.config.time_inf
            )

    # --------------------------------------------------------
    # One simulation time step
    # --------------------------------------------------------

    def step(self):

        dt = self.config.dt

        # 1. Brownian motion
        self.environment.brownian_motion(
            self.particles,
            dt,
            self.config.mysig
        )

        # 2. Masking behavior
        self.mask_model.update(self.particles)

        # 3. Update infection timers
        self.infection_model.update_infection_timers(
            self.particles
        )

        # 4. New infections
        self.infection_model.update(
            self.particles
        )

        # 5. Radial pull and rotation
        for particle in self.particles:

            self.environment.update_position(
                particle,
                dt
            )

        # 6. Update time
        self.t += dt

    # --------------------------------------------------------
    # Simulation statistics
    # --------------------------------------------------------

    def get_statistics(self):

        infected = sum(
            p.infected for p in self.particles
        )

        masked = sum(
            p.mask_on for p in self.particles
        )

        return {
            "time": self.t,
            "infected": infected,
            "masked": masked,
            "susceptible": self.config.numparts - infected
        }

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    def run(self):

        for ii in range(self.config.numouts):

            # Display current state
            self.visualizer.plot(
                self.particles,
                self.t
            )

            # Advance simulation
            for jj in range(self.config.numsteps):

                self.step()

            # Print progress
            if (ii + 1) % 100 == 0:

                stats = self.get_statistics()

                print(
                    f"Time: {stats['time']:.3f} | "
                    f"Infected: {stats['infected']} | "
                    f"Masked: {stats['masked']}"
                )

        # Final plot
        self.visualizer.plot(
            self.particles,
            self.t
        )

        self.visualizer.finish()
