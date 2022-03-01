# Kompressor

![GitHub](https://img.shields.io/github/license/rosalindfranklininstitute/kompressor?kill_cache=1)

| Branch  | CI | Coverage |
|:-:|:-:|:-:|
| `pr-adjust-readme-usage` (active) | [![Build](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=pr-adjust-readme-usage)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/pr-adjust-readme-usage/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |
| `main` | [![Build](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/main/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |
| `development`  | [![Build](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/development/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |

A neural compression framework built on top of JAX.

## Install

`setup.py` assumes a compatible version of JAX and JAXLib are already installed. Automated build is tested for a `cuda:11.5.1-cudnn8-devel-ubuntu20.04` environment with `jax-v0.3.1`.

A base docker image is available with CUDA and JAX installed in the `quay.io/rosalindfranklininstitute/jax:v0.3.1` Quay.io image.

## Install & Run through Docker environment

Docker image for the Kompressor dependencies are provided in the `quay.io/rosalindfranklininstitute/kompressor:<tag>` Quay.io image. Images are available for each branch of this repository (main, development, ect).

```
# Run the container for the Kompressor environment using the code for Kompressor installed in the image
docker run --rm quay.io/rosalindfranklininstitute/kompressor:<tag> \
    python -m pytest --cov=/usr/local/kompressor/src/kompressor /usr/local/kompressor/tests
```

```
# Run the container for the Kompressor environment using the code for Kompressor you have locally injected into the image
git clone https://github.com/rosalindfranklininstitute/kompressor.git
cd kompressor
docker run --rm -v $(pwd):/usr/local/kompressor -w /usr/local/kompressor \
    quay.io/rosalindfranklininstitute/kompressor:<tag> \
    python -m pytest --cov=/usr/local/kompressor/src/kompressor /usr/local/kompressor/tests
```

```
# Run the container for the Kompressor environment and host a jupyter lab server for using the example notebooks
git clone https://github.com/rosalindfranklininstitute/kompressor.git
cd kompressor
docker run --rm -v $(pwd):/usr/local/kompressor -w /usr/local/kompressor \
    quay.io/rosalindfranklininstitute/kompressor:<tag> \
    jupyter lab --port 8889 --no-browser --notebook-dir=/usr/local/kompressor
```

## Install & Run through Singularity environment

Singularity image for the Kompressor dependencies are provided in the `rosalindfranklininstitute/kompressor/kompressor:<tag>` cloud.sylabs.io image.

Use the `--nv` flag to enable CUDA GPUs.

```
# Pull an singularity image down from cloud.sylabs.io
singularity pull library://rosalindfranklininstitute/kompressor/kompressor:<tag>
```

```
# or... Build a singularity image from a docker image on Quay.io
singularity build kompressor_<tag>.sif docker://quay.io/rosalindfranklininstitute/kompressor:<tag>
```

```
# Run the container for the Kompressor environment using the code for Kompressor installed in the image
singularity run --nv kompressor_<tag>.sif \
    python -m pytest --cov=/usr/local/kompressor/src/kompressor /usr/local/kompressor/tests
```

```
# Run the container for the Kompressor environment using the code for Kompressor you have locally injected into the image
git clone https://github.com/rosalindfranklininstitute/kompressor.git
cd kompressor
singularity run --nv -B $(pwd):/usr/local/kompressor -W /usr/local/kompressor \
    kompressor_<tag>.sif \
    python -m pytest --cov=/usr/local/kompressor/src/kompressor /usr/local/kompressor/tests
```

```
# Run the container for the Kompressor environment and host a jupyter lab server for using the example notebooks
git clone https://github.com/rosalindfranklininstitute/kompressor.git
cd kompressor
singularity run --nv -B $(pwd):/usr/local/kompressor -W /usr/local/kompressor \
    kompressor_<tag>.sif \
    jupyter lab --port 8889 --no-browser --notebook-dir=/usr/local/kompressor
```