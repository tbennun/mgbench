#!/bin/sh
# MGBench: Multi-GPU Computing Benchmark Suite
# Copyright (c) 2016, Tal Ben-Nun
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of the copyright holders nor the names of its 
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

NUMGPUS=`./build/numgpus`
echo "Number of GPUs: ${NUMGPUS}"
if [ $NUMGPUS -eq 0 ]
then
    echo "No GPUs found, aborting test."
    exit 0
fi

# Run L1 tests
echo "L1 Tests"
echo "--------"

echo "1/5 Half-duplex (unidirectional) memory copy"
./build/halfduplex > l1-halfduplex.log

echo "2/5 Full-duplex (bidirectional) memory copy"
./build/fullduplex > l1-fullduplex.log

echo "3/5 Half-duplex DMA"
./build/uva > l1-uvahalf.log

echo "4/5 Full-duplex DMA"
./build/uva --fullduplex > l1-uvafull.log

echo "5/5 Scaling"
./build/sgemm -n 4096 -k 4096 -m 4096 --repetitions=100 --regression=false --scaling=true > l1-scaling.log

# Run L2 tests
echo ""
echo "L2 Tests"
echo "--------"

# Matrix multiplication
echo "1/5 Matrix multiplication (correctness)"
./build/sgemm -n 1024 -k 1024 -m 1024 --repetitions=100 --regression=true > l2-sgemm-correctness.log
echo "2/5 Matrix multiplication (performance)"
./build/sgemm -n 8192 -k 8192 -m 8192 --repetitions=100 --regression=false > l2-sgemm-perf.log

# Stencil operator
echo "3/5 Stencil (correctness)"
./build/gol --repetitions=5 --regression=true > l2-gol-correctness.log
echo "4/5 Stencil (performance)"
./build/gol --repetitions=1000 --regression=false > l2-gol-perf.log

# Test each GPU separately
echo "5/5 Stencil (single GPU correctness)"
echo "" > l2-gol-single.log
i=0
while [ $i -lt $NUMGPUS ]
do
    echo "GPU $i" >> l2-gol-single.log
    echo "===========" >> l2-gol-single.log
    ./build/gol --num_gpus=1 --repetitions=5 --regression=true --gpuoffset=$i >> l2-gol-single.log
    i=`expr $i + 1`
done

echo "Done!"
