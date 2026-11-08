
from dataclasses import dataclass


@dataclass
class SimulationConfig:

    # Population parameters
    numparts: int = 1000
    initial_infection_fraction: float = 0.01

    # Brownian motion
    mysig: float = 0.1

    # Infection parameters
    mindist0: float = 0.0007
    prob_inf0: float = 0.5
    time_inf: int = 100

    # Masking parameters
    mask_prob: float = 0.001
    mask_off_prob: float = 0.05
    mask_mult: float = 10.0
    mask_red: float = 0.1
    learning_dist: float = 0.1

    # Environment parameters
    R0: float = 1.0
    theta0: float = 0.0
    pull0: float = 0.1
    rotspd: float = 1.0

    # Time stepping
    dt: float = 1e-3
    numsteps: int = 10
    numouts: int = 1000

    # Random seed (None means random initialization)
    random_seed: int | None = None

    @property
    def numinf(self):
        return int(self.initial_infection_fraction * self.numparts)
