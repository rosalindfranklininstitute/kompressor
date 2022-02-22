# MIT License
#
# Copyright (c) 2020 Joss Whittle
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

# Install packages and register python3 as python
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install --no-install-recommends -y dialog apt-utils && \
    apt-get install --no-install-recommends -y build-essential git wget python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir --upgrade \
        numpy six wheel mock pytest pytest-cov PyYAML coverage

WORKDIR /usr/local/jax
RUN git clone --branch jax-v0.3.1 --depth 1 git@github.com:google/jax.git . && \
    python build/build.py  \
        --enable_cuda \
        --cuda_path='/usr/local/cuda' ` \
        --cudnn_path='/usr' ` \
        --cuda_version='11.5.1' ` \
        --cudnn_version='8.0.5' && \
    pip install --no-cache-dir --upgrade dist/*.whl && \
    pip install -e .

RUN pip install --no-cache-dir --upgrade \
        git+https://github.com/deepmind/dm-haiku && \
    pip install --no-cache-dir --upgrade \
        tensorflow tbp-nightly && \
    pip install --no-cache-dir --upgrade \
        jupyter jupyterlab && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

# Install Kompressor
ADD . /usr/local/kompressor
WORKDIR /usr/local/kompressor
RUN pip install -e . && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +