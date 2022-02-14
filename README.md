# Kompressor

![CI - Build Docker Image and Execute Tests](https://github.com/rosalindfranklininstitute/kompressor/workflows/CI%20-%20Build%20Images%20and%20Execute%20Tests/badge.svg) [![codecov](https://codecov.io/gh/rosalindfranklininstitute/kompressor/branch/master/graph/badge.svg?token=)](https://codecov.io/gh/rosalindfranklininstitute/kompressor)

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
