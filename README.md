# GeodesyNets

Code to train visualize and evaluate neural density fields using pytorch.

The code was developed and use for the writing of the paper:

Dario Izzo and Pablo Gomez, "Geodesy of irregular small bodies via neural density fields: geodesyNets". [arXiv:2105.13031](https://arxiv.org/pdf/2105.13031.pdf). (2021).

## Installation

We recommend using a [conda](https://docs.conda.io/en/latest/) environment to run this code. Once you have conda, (we also strongly suggest mamba istalled on the base environment) you can simply execute the `install.sh` script to create a conda environment called `geodesynet` with all required modules.

Note that to run some of the notebooks you may also need other dependencies.

## Architecture at a glance

A geodesyNet represents the body density directly as a function of Cartesian coordinates.
Recently, (see <https://github.com/bmild/nerf>)  a related architecture called Neural Radiance Fields (NeRF) was introduced to represent three-dimensional objects and complex scenes with an impressive accuracy learning from a set of two-dimensional images. The training of a NeRF solves the inverse problem of image rendering as it back-propagates the difference between images rendered from the network and a sparse set of observed images.

Similarly, the training of a geodesyNet solves the gravity inversion problem. The network learns from a dataset of measured gravitational accelerations back-propagating the difference to the corresponding accelerations computed from the density represented by the network.

The overall architecture to learn a neural density field is shown below:

![GeodesyNet Architecture](/figures/Fig1.png)

### Neural Density Field for 67p Churyumov-Gerasimenko

Once the network is trained we can explore and visualize the neural density field using techniques similar to 3D image scanning. This
results in videos such as the one below, obtained using the gravitational signature of the comet 67p Churyumov-Gerasimenko. Units are non dimensional.

![Neural Density Field for 67p](/figures/67p_low.gif)

### Neural Density Field for Bennu

Similarly, the video below refers to the results of differential training over a heterogenous Bennu model. Units are non dimensional.

## Investigation of the robustness of the neural density fields

### Overview

The results of the study and a detailed description are given in the following paper. It was presented as poster at ESA's GNC-ICATT 2023.

[Schuhmacher, J., Gratl, F., Izzo, D., & Gómez, P. (2023). Investigation of the Robustness of Neural Density Fields. arXiv preprint arXiv:2305.19698.](https://arxiv.org/abs/2305.19698)

The figure below gives a brief overview on the parameters that were investigated. The results are given in `<repo_root>/robustness_analysis/**`.

![Neural Density Field for 67p](/figures/investigation-of-the-robustness.png)

### Execution

You can run an experiment with a configuration from the `cfgs/robustness_analysis` folder by executing the `run_training.py` script. For example:

```bash
    python run_training.py <config.toml>  [optional_gpu_id]
```

The results will be stored in a `results` folder in this repository with a subfolder named after the `name` in the configuration file.
The training can be gracefully interrupted (the running training-run will still finish!) by sending `SIGTERM` to the process.

You can re-validate a trained model (in case you want to check the performance for a different set of constraints) by:

```bash
    python run_validation.py <path/to/folder-with-results>  [optional_gpu_id]
```

Please note that in the course of the code rebuild for this study, some Python modules are not integrated into the current code framework. This includes especially everything that is prefixed with `legacy*`.
