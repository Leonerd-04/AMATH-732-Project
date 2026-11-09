# README

# Particle-Based Epidemic Simulation

An agent-based simulation of an infectious disease spreading through a population of particles confined to a rotating 2D disk. Particles diffuse randomly, are pulled toward a preferred radius (with a strong angular trap), can become infected when close to neighbors, and can adopt or remove masks based on their own infection history.

---

## Overview

The simulation models a population of point particles moving in a 2D plane. Each timestep, particles:

1. Undergo Brownian (random) motion.
2. Update their mask-wearing state.
3. Count down infection timers.
4. May become newly infected.
5. Are pulled toward a preferred radius and rotated around the origin.

A live Matplotlib animation visualizes the state (unmasked, infected, masked) in real time.

---

## Repository Structure

```
.
├── simulation.py        # Main orchestrator: initialization, step loop, run loop
├── environment.py       # Spatial field: Brownian motion, radial pull, rotation
├── infection_model.py   # Disease transmission logic
├── mask_model.py        # Mask adoption/removal behavior
├── particle.py          # Particle data container (position, infection, mask)
├── visualizer.py        # Live Matplotlib visualization
└── README.md
```
---

## Components

### `Simulation`
The driver class. It:

- Seeds NumPy's RNG if `config.random_seed` is set.
- Initializes `numparts` particles in an annulus (radius `0.8–1.2`) at uniformly random angles.
- Randomly selects `numinf` initial infections (`replace=True`, so duplicates are possible) and calls `particle.infect(time_inf)` on each.
- Instantiates `Environment`, `InfectionModel`, `MaskModel`, and `Visualizer`.
- Runs a fixed step sequence in `step()` and repeats it `numouts × numsteps` times in `run()`.

**Step order (per `dt`):**
1. `Environment.brownian_motion`
2. `MaskModel.update`
3. `InfectionModel.update_infection_timers`
4. `InfectionModel.update` (new infections)
5. `Environment.update_position` (radial pull + rotation)
6. `t += dt`

Progress is printed every 100 outer iterations; a final plot is shown at the end.

### `Environment`
Defines the spatial "field."

- **`radial_pull(theta)`** — returns a restoring coefficient `-pull0 * (1 + 99·exp(-((theta-theta0)/0.4)²))`. This is a uniform pull toward radius `R0`, greatly amplified in a narrow angular band around `theta0` (a Gaussian "trap"/"gate").
- **`brownian_motion(particles, dt, sigma)`** — adds Gaussian noise of scale `sqrt(dt)·sigma` to each particle's `(x, y)`.
- **`update_position(particle, dt)`** — converts to polar coordinates, applies `r += dt·radial_pull(theta)·(r - R0)`, rotates `theta += rotspd·dt`, and converts back.

The combination produces particles that orbit the origin at `rotspd` while being held near radius `R0`, with a strong preference to stay near the `theta0` angular sector.

### `InfectionModel`
Handles transmission.

- **`update_infection_timers(particles)`** — decrements each particle's infection countdown (delegates to `Particle.update_infection_timer`).
- **`calculate_distances(particles)`** — builds the N×N pairwise Euclidean distance matrix via broadcasting; self-distances are set to `inf`.
- **`update(particles)`** — computes each particle's *minimum distance to any other particle* (regardless of that particle's infection status). A particle becomes newly infected if:
  - its nearest-neighbor distance is less than `mindist0`,
  - a uniform random draw is below its per-particle probability (`prob_inf0`, multiplied by `mask_red` if masked),
  - and it is not already infected.

New infections are determined **simultaneously** from the pre-step state, then applied — order-independent.

### `MaskModel`
Governs mask adoption and removal.

- Base probability of putting on a mask is `mask_prob` for all particles.
- If a particle has a stored sickness location `(x_sick, y_sick)` (set only when infected via `Particle.infect`) and its current distance to that location is less than `learning_dist`, its put-on probability is multiplied by `mask_mult`. This is the "learning" effect.
- `mask_off_prob` governs random removal.
- Unmasked particles may put on a mask; masked particles may take one off — each decided independently per step.

### `Particle`
A plain data container:

| Attribute | Meaning |
|---|---|
| `x`, `y` | Cartesian position |
| `infected` | Whether currently infected |
| `infection_timer` | Remaining timesteps of infection |
| `x_sick`, `y_sick` | Location where the particle was infected (only set on infection) |
| `mask_on` | Whether currently wearing a mask |

Methods:
- **`infect(time_inf)`** — sets `infected=True`, sets `infection_timer=time_inf`, records `(x_sick, y_sick)` at the current position.
- **`update_infection_timer()`** — decrements the timer; clears `infected` when it reaches zero (recovery).

### `Visualizer`
Live Matplotlib animation.

- All particles: small black dots.
- Infected particles: red circles.
- Masked particles: green stars.
- Fixed view `[-1.5, 1.5]²`, equal aspect, grid, title showing simulation time.
- `finish()` disables interactive mode and calls `plt.show()`.

---

## Configuration

`Simulation` and its components read a shared `config` object. The attributes referenced by the code are:

| Attribute | Used by | Meaning |
|---|---|---|
| `numparts` | Simulation | Number of particles |
| `numinf` | Simulation | Number of initial infections |
| `random_seed` | Simulation | RNG seed (or `None`) |
| `dt` | Simulation, Environment | Timestep |
| `time_inf` | Simulation, InfectionModel, Particle | Infection duration (steps) |
| `mysig` | Simulation → Environment | Brownian noise scale |
| `numouts` | Simulation | Outer loop iterations (plot intervals) |
| `numsteps` | Simulation | Inner steps per outer iteration |
| `R0` | Environment | Preferred radius |
| `theta0` | Environment | Angular trap center |
| `pull0` | Environment | Radial pull strength |
| `rotspd` | Environment | Angular rotation speed |
| `mindist0` | InfectionModel | Infection distance threshold |
| `prob_inf0` | InfectionModel | Base per-step infection probability |
| `mask_red` | InfectionModel | Probability multiplier when masked |
| `mask_prob` | MaskModel | Base probability of putting on a mask |
| `mask_off_prob` | MaskModel | Probability of taking a mask off |
| `mask_mult` | MaskModel | Multiplier on put-on probability near sickness location |
| `learning_dist` | MaskModel | Distance within which the learning effect applies |

---

## Usage

```python
from simulation import Simulation

class Config:
    random_seed = 42
    numparts = 500
    numinf = 5
    dt = 0.01
    time_inf = 100
    mysig = 0.05
    numouts = 200
    numsteps = 10

    R0 = 1.0
    theta0 = 0.0
    pull0 = 1.0
    rotspd = 0.1

    mindist0 = 0.05
    prob_inf0 = 0.01
    mask_red = 0.5

    mask_prob = 0.001
    mask_off_prob = 0.0005
    mask_mult = 10.0
    learning_dist = 0.2

sim = Simulation(Config())
sim.run()
```

See main for yet another example.
---

## Dependencies

- Python 3
- NumPy
- Matplotlib

Install with:

```bash
pip install numpy matplotlib
```

---

## Known Modeling Characteristics

These are properties of the code **as written**:

- **Density-driven transmission.** `InfectionModel.update` uses the distance to the *nearest particle of any kind*, not the nearest *infected* particle. There is no check that a nearby neighbor is actually sick. The epidemic therefore spreads based on local density plus a random draw, rather than through explicit contacts with infected individuals.
- **SIS-like dynamics.** Recovered particles (`infection_timer` reaching 0) become susceptible again; there is no lasting immunity.
- **Duplicate initial infections.** `np.random.choice(n, numinf, replace=True)` may pick the same particle more than once, so the effective number of initially infected particles can be less than `numinf`.
- **Mask "learning" applies only to previously infected particles.** `x_sick`/`y_sick` are set only in `Particle.infect`, so only particles that have been infected carry a sickness location to learn from.
- **Simultaneous infection updates.** New infections within a step are computed from the pre-step state and applied together, avoiding order dependence.
- **Fixed plot window.** The visualizer uses a hard-coded `[-1.5, 1.5]²` view; particles that drift outside this range will not be visible (though the radial pull keeps them near `R0`).

---

