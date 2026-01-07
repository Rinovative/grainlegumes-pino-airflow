# GrainLegumes-PINOs: Physics-Informed Neural Operators for Porous Media Flow  
### *Specialization Project (VP1) – MSE Data Science, Autumn 2025*

**Master of Science in Engineering – Major Data Science**  
**Eastern Switzerland University of Applied Sciences (OST)**  
**Author:** Rino M. Albertin  
**Supervisor:** Prof. Dr. Christoph Würsch  

---

## 📌 Project Overview

This specialization project studies the learning of physically consistent surrogate models for incompressible air flow in **porous granular media** using **Physics-Informed Neural Operators (PINOs)**.

High-fidelity permeability and porosity fields are synthetically generated in **MATLAB** and simulated with **COMSOL Multiphysics** using a Darcy–Brinkman formulation.  
The central objective is to train two-dimensional neural operators that learn the operator mapping

  **(κ, φ, p_bc) → (p, u, v)**

from spatially varying permeability tensors κ, porosity fields φ, and inlet pressure boundary conditions p_bc to pressure and velocity fields, while explicitly enforcing physical consistency through PDE-based constraints.

The repository provides a complete, modular research pipeline covering:
- 🧩 **Physics-based data generation**  
  A fully automated MATLAB-driven pipeline for synthetic porous-media data generation, including:
  - **Parameter sampling**: space-filling sampling strategies (uniform, LHS, Sobol)
  - **Structure synthesis**: stochastic multi-scale structure field generation as latent geometric backbone
  - **Permeability construction**: physically consistent mapping to scalar and tensor-valued permeability fields
  - **Porosity modelling**: independent porosity field generation with global Kozeny–Carman level anchoring
  - **Boundary conditions**: low-dimensional, spatially varying inlet pressure boundary conditions
  - **High-fidelity simulation**: batch-controlled Darcy–Brinkman simulations in COMSOL via LiveLink for MATLAB  
  The pipeline supports resume-safe batch execution, reproducible seeding, and rich data export (CSV + JSON).

- 📊 **Exploratory Data Analysis (EDA)**  
  An interactive EDA framework including:
  - **Statistical analysis**: case-level distributions of generator parameters, meta statistics, and reduced field statistics (min/mean/max)
  - **Spectral analysis**: two-dimensional FFT-based analysis
  - **Scale diagnostics**: isotropic radial energy spectra and vertical spectral evolution analysis

- ⚙️ **Neural Operator training (FNO / U-NO / PINOs)**  
  A modular, reproducible training framework for neural operator models, including:
  - **Architectures**: FNO, U-NO, and physics-informed variants (PI-FNO, PI-U-NO)
  - **Multi-field I/O**: spatial coordinates, tensor-valued permeability, porosity, inlet pressure → velocity components and pressure
  - **Physics-informed learning**: COMSOL-consistent Brinkman PINO loss combining data fidelity and PDE residuals
  - **Spectral diagnostics**: optional non-intrusive forward hooks on spectral convolution layers
  - **Experiment tracking**: structured logging with full model, optimizer, scheduler, and loss configurations (wandb + checkpoints)  

- 🧪 **High-fidelity evaluation framework**  
  A full scientific evaluation suite for systematic model comparison and assessment, supporting both cross-model comparison on fixed datasets and cross-dataset generalisation analysis (ID and OOD), including:
  - **Global error analysis**: L2 and relative L2 metrics, distributions, CDFs, mean and standard-deviation error maps, and frequency-domain error spectra  
  - **Error decomposition**: error vs output magnitude and error vs distance to domain boundaries  
  - **Physical consistency checks**: velocity divergence, mass conservation error maps, pressure boundary-condition consistency, and full Darcy–Brinkman operator residual evaluation  
  - **Error sensitivity analysis**: parameter–error correlation heatmaps and parameter-wise error trend analysis  
  - **Interactive sample inspection**: multi-field prediction, ground truth, and error viewer with permeability field visualisation  
  - **Outlier and extreme-case analysis**: worst-case per output channel and extreme input parameter multi-field case viewer

- 🧬 **Interactive research environment**  
  All evaluation components are provided as interactive Jupyter widgets with dataset selection, case sliders and dynamic plots for systematic exploration of model behaviour.

---

## 🧭 Data Flow Overview

<details>
<summary><strong>Expand</strong></summary>

```mermaid
flowchart TD

A1[Parameter sampling  
Uniform / LHS / Sobol]

--> A2[Structure synthesis  
Multi-scale stochastic geometry]

--> A3[Permeability construction  
Scalar & tensor fields κ]

A2 --> A4[Porosity modelling  
Field φ with Kozeny–Carman anchoring]
A3 --> A4

A4 --> A5[Boundary condition generation  
Low-dimensional p_bc fields]

A5 --> A6[Export generator outputs  
Metadata (JSON)  
Fields (CSV)]

A6 --> B1[COMSOL Multiphysics  
Import generator outputs]

B1 --> B2[Brinkman flow solver  
Compute p, u, v fields]

B2 --> B3[Export simulation outputs  
Fields (CSV)  
Metadata (JSON)]

B3 --> C1[Load CSV + JSON  
Prune & compress metadata]

C1 --> C2[Tensor construction  
Normalize & transform fields]

C2 --> C3[Automatic channel pruning]

C3 --> C4[Build per-case files  
Structured PyTorch case (.pt)]

C4 --> D1[Build final dataset  
Training-ready PyTorch (.pt)
Merge all cases  
Stack tensors]

C4 --> D2[Exploratory Data Analysis  
Statistical & spectral validation]

D1 --> F[Neural Operator Training  
FNO / U-NO / PINO  
Physics-informed learning]

F --> G[Scientific Evaluation  
Error analysis  
Physical consistency  
Sensitivity & outlier studies]
```

</details>

---

## ⚙️ Local Execution

<details>
<summary><strong>Option A – Run in Visual Studio Code with Docker Dev Container (recommended)</strong></summary>

**Requirements**
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Visual Studio Code](https://code.visualstudio.com/)
- VS Code extension **“Dev Containers”**

**Steps**
```bash
git clone https://github.com/Rinovative/grainlegumes-pino.git
cd grainlegumes-pino
```
1. Open the folder in VS Code  
2. Reopen in Container (via prompt or `F1 → Dev Containers: Reopen in Container`)  
3. Launch one of the notebooks or trainingsskripts  

</details>

<details>
<summary><strong>Option B – Run via Docker CLI (without VS Code)</strong></summary>

```bash
git clone https://github.com/Rinovative/grainlegumes-pino.git
cd grainlegumes-pino

docker build -t pino-dev .
docker run -it --rm -p 8888:8888 -v $(pwd):/app pino-dev
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

Then open the URL shown in the terminal.

</details>

---

## 📂 Repository Structure
<details>
<summary><strong>Show project tree</strong></summary>

```bash
.
├── .devcontainer/                                      # VS Code Dev Container configuration
│   └── devcontainer.json                               # Container setup and environment definition
│
├── data/                                               # Final trained modelss and batch training datasets
│   ├── processed/                                      # Final trained models
│   └── raw/                                            # COMSOL output and metadata for batch before preprocessing
│       ├── samples_uniform_var10_N1000/                # Example batch of simulation cases
│       │   ├── cases/                                  # Individual case files with (κ, p, U)
│       │   └── meta.pt                                 # Batch generation parameters
│       └── ...                                         
│
├── data_generation/                                    # MATLAB → COMSOL → PyTorch data creation pipeline
│   ├── comsol/                                         # COMSOL model templates for automated simulation
│   │   ├── template_brinkman.mph                       # Base Brinkman model file
│   │   ├── template_brinkman_cluster.mph               # Cluster version
│   │   └── template_brinkman_tensor.mph                # Tensor variant for permeability field
│   │
│   ├── data/                                           # Generated datasets
│   │   ├── meta/                                       # Metadata describing batch
│   │   │   ├── samples_uniform_var10_N1000.csv         # Generation parameters for cases of batch
│   │   │   ├── samples_uniform_var10_N1000.json        # Metadata for batch generation
│   │   │   └── ...                                     
│   │   │
│   │   ├── processed/                                  # COMSOL outputs
│   │   │   ├── samples_uniform_var10_N1000/            # Processed dataset directory
│   │   │   │   ├── case_0001_sol.csv                   # Example processed field solution
│   │   │   │   └── ...                                 
│   │   │   └── ...                                     
│   │   │
│   │   └── raw/                                        # MATLAB permability-field
│   │       ├── samples_uniform_var10_N1000/            # Individual batch
│   │       │   ├── case_0001.csv                       # Raw permeability field data
│   │       │   ├── case_0001.json                      # Associated metadata for this case
│   │       │   └── ...                                 
│   │       └── ...                                     
│   │
│   └── matlab/                                         # MATLAB scripts for permeability generation and COMSOL coupling
│       ├── functions/                                  # Modularized MATLAB functions
│       │   ├── core/                                   # Core utilities for data generation and visualization
│       │   │   ├── gen_permeability.m                  # Generates synthetic permeability fields κ(x)
│       │   │   ├── run_comsol_case.m                   # Executes a single COMSOL simulation case
│       │   │   ├── sample_parameters.m                 # Creates randomized parameter sets for DoE
│       │   │   └── visualize_case.m                    # Visualization helper for MATLAB/COMSOL outputs
│       │   │
│       │   └── test/                                   # MATLAB test routines for validation
│       │       ├── test_generate_permeability_fields.m # Test for permeability generation
│       │       ├── test_run_comsol_case.m              # Test for COMSOL automation routine
│       │       └── test_visualize_case.m               # Test for visualization and output integrity
│       │
│       ├── batch_run.m                                 # Batch execution for full dataset generation
│       ├── build_batch_dataset.py                      # Python converter for merging raw COMSOL outputs into .pt
│       ├── merge_batch_cases.py                        # Combines multiple cases into unified datasets
│       ├── permeability_field_viewer.mlx               # MATLAB Live Script for permeability-field inspection
│       └── singel_run.m                                # Single test run for debugging and prototyping
│   
├── docs/                                               # Project documentation, plots, and figures
│
├── model_training/                                     # Core training and analysis environment
│   ├── data/                                           # Training datasets and model checkpoints
│   │   ├── meta/                                       # 
│   │   ├── processed/                                  # 
│   │   └── raw/                                        # Merged datasets used as input
│   │       ├── samples_uniform_var10_N1000/            # Example batch
│   │       │   ├── meta.pt                             # Batch generation parameters
│   │       │   └── samples_uniform_var10_N1000.pt      # Main training tensor data
│   │       └── ...                                     
│   │
│   ├── notebooks/                                      # Interactive notebooks for analysis and visualization
│   │   └── EDA.ipynb                                   # Exploratory Data Analysis for PINO input fields
│   │
│   ├── src/                                            
│   │   ├── eda/                                        # Spectral and statistical analysis utilities
│   │   │   ├── __init__.py                             
│   │   │   └── eda_spectral_analysis.py                # Main EDA routines for PSD and field spectra
│   │   │
│   │   ├── model/                                      # 
│   │   │   ├── __init__.py                             
│   │   │   └── XXX.py                                  #
│   │   │
│   │   └── util/                                       # Shared helper functions
│   │       ├── __init__.py                             
│   │       ├── util_data.py                            # Data loading and preprocessing routines
│   │       └── util_nb.py                              # Notebook utilities (visualization, widgets)
│   │
│   └── train_pino.py                                   # Main training entry script for PINO
│
├── .dockerignore                                       # Docker build exclusion list
├── .gitignore                                          # Git exclusion list
├── Dockerfile                                          # Docker image setup for reproducible environment
├── environment.yml                                     # Conda/Mamba environment specification
├── pyproject.toml                                      # Poetry configuration for dependencies
└── README.md                                           # Project overview and documentation
```
</details>

---

## 🧠 Methodology

1. **Data Generation (MATLAB + COMSOL)**  
   Random κ fields are generated in MATLAB and solved for p and U in COMSOL (Brinkman flow).  
2. **Data Preparation (Python)**  
   Case files and metadata are merged into structured `.pt` datasets.  
3. **Exploratory Data Analysis (EDA)**  
   Statistical and spectral inspection of fields using Matplotlib and ipywidgets.  
4. **Model Training (PINO)**  
   Train a Fourier-based Physics-Informed Neural Operator to learn the mapping κ → (p, U).  
5. **Evaluation and Diagnostics**  
   Visualize residual loss, convergence curves, and spectral error maps.

---

## 📊 Visualizations

---

## 📄 License

This project is released under the []().

---

## 📚 Reference

Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiaffini, M., Duruisseaux, V., George, R. J., Bonev, B., Azizzadenesheli, K., Berner, J., & Anandkumar, A. (2025).  
*A Library for Learning Neural Operators.*  
*arXiv preprint* [arXiv:2412.10354](https://arxiv.org/abs/2412.10354)