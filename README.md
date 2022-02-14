# Kompressor

![GitHub](https://img.shields.io/github/license/rosalindfranklininstitute/kompressor?kill_cache=1)

| Branch  | CI | Coverage |
|:-:|:-:|:-:|
| `development` (active) | [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/development/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |
| `main` | [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/main/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |
| `development`  | [![GitHub Workflow Status (branch)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rosalindfranklininstitute/kompressor/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/development/graph/badge.svg?token=nJk2eue86w)](https://codecov.io/gh/rosalindfranklininstitute/kompressor) |

A neural compression framework built on top of JAX.

## Install

```
git clone https://github.com/rosalindfranklininstitute/kompressor.git
cd kompressor
pip install -e .

# Run tests
python -m pytest --cov=src/kompressor tests/
```

## Install & Run through Docker environment

Docker image for the Kompressor dependencies are provided in the `rosalindfranklininstitute/kompressor:master` Quay.io image.

```
# Run the container for the Kompressor environment (mounting the present working directory)
docker run --rm rosalindfranklininstitute/kompressor:master \
    python -m pytest --cov=/usr/local/kompressor/src/kompressor /usr/local/kompressor/tests
```