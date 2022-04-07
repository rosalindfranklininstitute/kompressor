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


FROM quay.io/rosalindfranklininstitute/jax:v0.3.1

# Install git
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install --no-install-recommends -y dialog apt-utils && \
    apt-get install --no-install-recommends -y git && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade \
        git+https://github.com/deepmind/dm-haiku && \
    pip install --no-cache-dir --upgrade \
        tensorflow tbp-nightly && \
    pip install --no-cache-dir --upgrade \
        jupyter jupyterlab && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

# Install Kompressor dependencies to enable caching
ADD requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

# Install Kompressor
ADD . /usr/local/kompressor
WORKDIR /usr/local/kompressor
RUN pip install -e . && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +
