# BSW Data Generation

Generate synthetic hysteresis-loop data for single-grain magnetic cubes using micromagnetic simulations via the [MaMMoS](https://github.com/MaMMoS-project) software stack.

## Overview

This project automates the generation of a dataset that maps **intrinsic** magnetic material parameters to **extrinsic** magnetic properties for a 50 nm single-grain cube geometry. Each simulation:

1. Randomly samples intrinsic material parameters (saturation magnetisation $M_s$, exchange stiffness $A$, and uniaxial anisotropy $K_1$) from physically meaningful ranges.
2. Runs a micromagnetic hysteresis-loop simulation using `mammos_mumag`.
3. Extracts the extrinsic properties (coercivity $H_c$, remanence $M_r$, and maximum energy product $(BH)_{\max}$) from the simulated loop using `mammos_analysis`.
4. Saves the results as ontology-annotated MaMMoS YAML files.

The collected dataset is stored in a single CSV file for analysis or machine-learning workflows.

## Parameter Ranges

| Parameter | Symbol | Range | Unit |
|---|---|---|---|
| Saturation Polarisation | $J_s$ | 0.1 – 5 | T |
| Exchange stiffness | $A$ | $10^{-13}$ – $10^{-11}$ | J/m |
| Uniaxial anisotropy | $K_1$ | $10^{4}$ – $10^{7}$ | J/m³ |

Samples where the exchange length $l_A = \sqrt{2A / (\mu_0 M_s^2)}$ or domain wall thickness $l_K = \sqrt{A / K_1}$ falls below 1 nm are rejected and re-sampled to ensure numerically stable simulations.

## Prerequisites

- [Pixi](https://pixi.sh) package manager
- Linux x86-64 (the environment is pinned to `linux-64`)

## Installation

Clone the repository and let Pixi resolve the environment:

```bash
git clone https://github.com/MaMMoS-project/BSW_data_generation.git
cd BSW_data_generation
pixi install
```

This installs all dependencies defined in `pixi.toml`, including `mammos`, `esys-escript`, and `jax` (with CUDA 12 support).

## Running Simulations

### Single simulation (local)

Run a single simulation on the current machine:

```bash
pixi run single-sim
```

This executes `generate_data.py`, which:

1. Creates a working directory under `data/<job_id>/`.
2. Samples random material parameters.
3. Runs the hysteresis-loop simulation.
4. Saves a `parameters.yaml` file with both intrinsic inputs and computed extrinsic outputs.
5. Cleans up intermediate simulation files (keeping only `.csv` data and metadata).

During the hysteresis loop, if the magnetisation does not flip direction by -10 T, the simulation is terminated early to save time, and the results are recorded as NaNs.

### Example batch simulations on SLURM (HPC)

Submit an array job to a SLURM cluster using the provided submission script:

```bash
sbatch --array=0-<N> submit_ada.sh
```

The script (`submit_ada.sh`) is configured for the `p.ada` partition with 4× A100 GPUs per node. Each SLURM task runs one independent simulation, and `srun` maps each task to a separate GPU via `CUDA_VISIBLE_DEVICES`. Logs are written to `logs/`.

Key SLURM settings (edit `submit_ada.sh` as needed):

| Setting | Value |
|---|---|
| Partition | `p.ada` |
| GPUs per node | 4× A100 |
| Tasks per node | 4 |
| CPUs per task | 18 |
| Wall time | 4 hours |

## Collecting Results

After simulations have completed, aggregate all individual `parameters.yaml` files into a single CSV:

```bash
pixi run collect-data
```

This executes `collect_data.py`, which scans `data/**/parameters.yaml`, concatenates the results, and writes `full.csv` — an ontology-annotated MaMMoS CSV containing one row per simulation with the columns:

| Column | Description | Unit |
|---|---|---|
| `Ms` | Saturation magnetisation | A/m |
| `A` | Exchange stiffness constant | J/m |
| `K1` | Uniaxial anisotropy constant | J/m³ |
| `D` | Demagnetising factor | — |
| `Hc` | Coercivity | A/m |
| `Mr` | Remanence | A/m |
| `BHmax` | Maximum energy product | N/m² |
| `filepath` | Path to the source `parameters.yaml` | — |

If any simulations failed or were terminated early, their extrinsic properties will be recorded as NaNs in the dataset. Practically this looks like an empty entry in the CSV, for Hc, Mr, and BHmax.
