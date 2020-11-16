# Kompressor

[![Build Status](https://travis-ci.com/JossWhittle/Kompressor.svg?token=vubJNeXp5jXw6jqyAo97&branch=master)](https://travis-ci.com/JossWhittle/Kompressor) [![Coverage Status](https://coveralls.io/repos/github/JossWhittle/Kompressor/badge.svg?branch=development&service=github&kill_cache=1)](https://coveralls.io/github/JossWhittle/Kompressor?branch=development&service=github&kill_cache=1)

A neural compression framework built on top of JAX.

## Install

```
git clone https://github.com/JossWhittle/Kompressor.git
cd Kompressor
pip install -e .

# Run tests
python -m pytest tests/*
```

## Install & Run through Docker environment

Docker image for the Kompressor dependencies (CPU execution only) are provided in the `josswhittle/kompressor:env-v1` Dockerhub image.

This image is used by the Travis CI builds.

```
git clone https://github.com/JossWhittle/Kompressor.git
cd Kompressor

# Run the container for the Kompressor environment (mounting the present working directory)
docker run -itd --name env -v $(pwd):/tmp/repo/ -w /tmp/repo/ josswhittle/kompressor:env-v1

# Install the current version version of Kompressor inside the container 
# Installation only persists until container is stopped
docker exec env pip install -e .

# Run tests (including code coverage)
docker exec env python -m pytest --cov=src/kompressor tests/

# Halt the container when finished
docker stop env
```
