# Kompressor

![GitHub](https://img.shields.io/github/license/rosalindfranklininstitute/kompressor?kill_cache=1)

| Branch  | CI | Coverage |
|:-:|:-:|:-:|
| `development` (active) | [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/development/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |
| `main` | [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/main/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |
| `development`  | [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/development/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |

| Release  | CI | Coverage |
|:-:|:-:|:-:|
| `v0.0.0` | [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/release.yml/badge.svg?branch=v0.0.0)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/v0.0.0/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |

A neural compression framework built on top of JAX.

## Install

`setup.py` assumes a compatible version of JAX and JAXLib are already installed. Automated build is tested for a `cuda:11.1-cudnn8-runtime-ubuntu20.04` environment with `jaxlib==0.1.76+cuda11.cudnn82`.

```
git clone https://github.com/rosalindfranklininstitute/kompressor.git
cd kompressor
pip install -e .

# Run tests
python -m pytest --cov=src/kompressor tests/
```

## Install & Run through Docker environment

Docker image for the Kompressor dependencies are provided in the `quay.io/rosalindfranklininstitute/kompressor:main` Quay.io image.

```
# Run the container for the Kompressor environment
docker run --rm quay.io/rosalindfranklininstitute/kompressor:main \
    python -m pytest --cov=/usr/local/kompressor/src/kompressor /usr/local/kompressor/tests
```

## Install & Run through Singularity environment

Singularity image for the Kompressor dependencies are provided in the `rosalindfranklininstitute/kompressor/kompressor:main` cloud.sylabs.io image.

```
singularity pull library://rosalindfranklininstitute/kompressor/kompressor:main
singularity run kompressor_main.sif \
    python -m pytest --cov=/usr/local/kompressor/src/kompressor /usr/local/kompressor/tests
```