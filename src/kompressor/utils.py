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


def yield_chunks(max_value, chunk):
    # Assert max value is positive
    assert max_value > 0

    # Assert chunk size is valid
    assert chunk > 3

    if chunk >= max_value:
        # If we can process in a single chunk than yield that chunk with no padding
        yield (0, max_value), (0, 0)

    else:
        # Yield a set of constant sized chunks along one axis including boundary conditions
        for idx in range(0, max_value, (chunk-3)):
            # Far edge of chunk, clamped against the right edge
            i1 = min(max_value, (idx+(chunk-2)))

            # Does this chunk border on the rightmost edge
            last = (i1 == max_value)

            # Near edge of chunk, constraining that every chunk must be of constant size
            i0 = max(0, ((i1-(chunk-2)) if last else idx))

            # Does this chunk border on the left most edge
            first = (i0 == 0)

            # Calculate constant width padding
            p0 = 0 if first else (2 if last  else 1)
            p1 = 0 if last  else (2 if first else 1)

            # Assert singleton chunk was handled by the other if-branch
            assert not (first and last)

            # Assert that the total padding was length 2 to ensure constant sized chunks
            assert (p0 + p1) == 2

            # Yield the chunk and the dynamic padding
            yield (i0, i1), (p0, p1)

            # Prevent duplicating last chunk
            if last:
                break


def validate_padding(padding):
    # Assert valid padding
    assert isinstance(padding, int)
    assert padding >= 0
