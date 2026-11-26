
# Opinion Dynamics on Counter-Rotating Annuli

This repository contains a Python-based agent-based simulation of opinion dynamics in two counter-rotating annular populations.

Individuals move stochastically within two orbiting populations and interact with nearby individuals, adjusting their opinions according to an interaction rule incorporating an **echo chamber effect**. The model explores how individual movement, local interactions, and opinion reinforcement influence the emergence of opinion polarisation and spatial segregation.

## Features

- **Agent-based modelling:** Simulates the movement and opinion evolution of large populations of individuals.
- **Counter-rotating annuli:** Models two spatially distinct populations moving in opposite directions.
- **Opinion dynamics:** Implements local opinion exchange with an adjustable echo effect.
- **Spatial segregation analysis:** Measures clustering, nearest-neighbour agreement, and angular opinion boundaries.
- **Data visualisation:** Generates plots tracking opinion distributions, polarisation, and spatial patterns over time.

## Repository Structure

```text
.
├── individual.py       # Individual agent behaviour
├── main.py             # Main simulation
├── plots.py            # Simulation analysis and visualisation
└── results/            # Simulation output and generated plots
```

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib
- Pandas

## Running the Simulation

Run the simulation using:

```bash
python main.py
```

Simulation data is saved in the `results/` directory, including:

- `individual_infections.csv`: Records individual opinions, positions, and other agent properties over time.
- `new_infections.csv`: Records newly acquired non-zero opinions during the simulation.

To generate the analysis plots, run:

```bash
python plots.py
```

The generated plots are saved in the `results/plots/` directory.

## Understanding the Plots

Below, I explain the different plots. Feel free to ask me if something does not make sense!

### [I] Spatial Snapshots

The spatial snapshots show how positive and negative opinions are distributed throughout the two interacting circles over time.

Each panel represents a different simulation timestep, with red indicating positive opinions, blue indicating negative opinions, and black representing individuals with a zero opinion.

The continuous colour scale illustrates the strength of each individual's opinion, with stronger positive and negative opinions represented by more intense colours. This allows changes in both spatial distribution and opinion intensity to be observed.

### [II] Distribution of Opinions

The histograms show the distribution of non-zero opinion strengths at five different points in the simulation.

Individuals with zero opinions are excluded from these histograms. The distributions illustrate how the balance between positive and negative opinions changes over time, as well as whether opinions tend to become stronger, weaker, or more concentrated around particular values.

### [III] Phase Portrait

The phase portrait tracks the number of positive ($N_+$) and negative ($N_-$) individuals throughout the simulation.

Each point represents the population state at a particular timestep, with the trajectory showing how the balance between the two competing opinions evolves. The green and red markers indicate the initial and final population states, respectively.

Individuals with zero opinions are not included in either count.

### [IV] Echo-Silo Diagnostics: Segregation Time Series

This figure examines whether individuals preferentially interact with others who share their opinion. The analysis considers only individuals with non-zero opinions and uses their spatial positions to identify local neighbourhoods.

It consists of three panels:

1. **k-NN segregation:** Measures the proportion of an individual's eight nearest opinionated neighbours who hold the same opinion. Higher values indicate stronger clustering of like-minded individuals. The dotted lines represent the expected same-opinion fractions under random mixing, based on the overall balance of positive and negative opinions in the population.

2. **Segregation excess:** Measures the difference between the observed local same-opinion neighbour fraction and its expected value under random mixing. Positive values indicate segregation, while negative values indicate greater interaction with opposing opinions than would be expected randomly.

3. **Normalised opinion interface:** Measures the proportion of nearby opinionated pairs that hold opposing opinions. Two individuals are considered nearby if their Euclidean distance is less than or equal to the specified threshold (0.05). Higher values indicate more contact between competing opinion groups, whereas lower values suggest greater spatial separation.

### [V] Angular Segregation

This figure examines how positive and negative opinions are distributed around each circle over time.

Each circle is divided into 36 angular segments. The heatmaps show the mean opinion strength of the non-zero-opinion individuals within each segment: red represents positive mean opinions, blue represents negative mean opinions, and grey indicates empty segments.

This reveals how opinions cluster spatially and whether distinct regions of opposing opinions develop.

The accompanying boundary plots count transitions between sufficiently close occupied angular segments with opposing majority signs. Empty angular segments are allowed between occupied segments when calculating these transitions, provided the gap does not exceed the specified maximum (currently one empty segment).

A higher number of boundaries indicates more frequent changes between positive- and negative-dominated regions, while a lower number suggests larger, more consolidated regions of similar opinion.

Together, these plots show how spatial opinion segregation develops independently within each circle.
