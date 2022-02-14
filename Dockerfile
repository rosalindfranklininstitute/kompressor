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


FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

# Install packages and register python3 as python
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install -y dialog apt-utils && \
    apt-get install -y build-essential git wget python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 install --no-cache-dir --upgrade \
        mock pytest pytest-cov PyYAML coverage \
        tqdm numpy scipy h5py pandas matplotlib && \
    pip3 install --no-cache-dir --upgrade \
        jax jaxlib==0.1.76+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
    pip3 install --no-cache-dir --upgrade \
        git+https://github.com/deepmind/dm-haiku && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

# Install Kompressor
ADD . /usr/local/kompressor
WORKDIR /usr/local/kompressor
RUN pip3 install -e . && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +