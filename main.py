
from config import SimulationConfig
from simulation import Simulation


def main():

    # Define simulation parameters
    config = SimulationConfig(

        # Population
        numparts=500,
        initial_infection_fraction=0.01,

        # Brownian motion
        mysig=0.1,

        # Infection
        mindist0=0.0007,
        prob_inf0=0.5,
        time_inf=100,

        # Masking
        mask_prob=0.001,
        mask_off_prob=0.05,
        mask_mult=10,
        mask_red=0.1,
        learning_dist=0.1,

        # Environment
        R0=1.0,
        theta0=0.0,
        pull0=0.1,
        rotspd=1.0,

        # Time stepping
        dt=1e-3,
        numsteps=10,
        numouts=1000,

        # Reproducibility
        random_seed=42
    )

    # Create and run simulation
    simulation = Simulation(config)

    simulation.run()


if __name__ == "__main__":
    main()
