# Directory structure

This directory contains the configuration files for the robustness analysis.

## Overview

```
.
├── README.MD
├── polyhedral_total_churyumov-gerasimenko.toml
├── polyhedral_total_eros.toml
├── pretraining_experiment_churyumov-gerasimenko
│   ├── res001
│   ├── res010
│   ├── res0_1
│   └── short
├── pretraining_experiment_eros
│   ├── res001
│   ├── res010
│   ├── res0_1
│   └── short
└── unused
```

## Contents

- `polyhedral_total_churyumov-gerasimenko.toml`: Configuration file for the robustness analysis of the
  Churyumov-Gerasimenko comet.
  Contains the extensive experiment with different kinds of perturbations,
  varying sample domain and ground-truth models.
- `polyhedral_total_eros.toml`: Configuration file for the robustness analysis of the Eros asteroid.
  Contains the extensive experiment with different kinds of perturbations,
  varying sample domain and ground-truth models.
- `pretraining_experiment_churyumov-gerasimenko`: Directory containing the results of the pretraining & fine-tuning
  experiment for the Churyumov-Gerasimenko comet. Further it varies the resolution and amount of training iterations.
    - `res001`: Results of the experiment with a resolution of 1%
    - `res010`: Results of the experiment with a resolution of 10%
    - `res0_1`: Results of the experiment with a resolution of 0.1%
    - `short`: Results of the experiment with a short training time (fine-tuning only without prior pretraining)
- `pretraining_experiment_eros`: ... for Eros
- `unused`: Directory containing the configuration file for the analysis of impact of the integration points of the
  Churyumov-Gerasimenko comet. This file is not used in the final analysis.