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

#######################################
# Find nvidia-smi for temperature tests
TEMPTEST=0
NVSMI=`which nvidia-smi`
if ! [ -x "$NVSMI" ]
then
    NVSMI=`find /usr/local -name 'nvidia-smi' 2> /dev/null`
    if ! [ -x "$NVSMI" ]
    then
        NVSMI=`find -L /etc -name 'nvidia-smi' 2> /dev/null`
        if ! [ -x "$NVSMI" ]
        then
            echo "WARNING: nvidia-smi not found"
        else
            TEMPTEST=1
        fi
    else
        TEMPTEST=1
    fi
else
    TEMPTEST=1
fi

if [ $TEMPTEST -eq 1 ]
then
    echo "Found nvidia-smi at ${NVSMI}"
fi
#######################################


# Run L0 diagnostics
echo ""
echo "L0 diagnostics"
echo "--------------"

echo "1/2 Computer information"
echo "CPU Info:" > l0-info.log
cat /proc/cpuinfo >> l0-info.log
echo "Memory Info:" >> l0-info.log
cat /proc/meminfo >> l0-info.log

echo "2/2 Device information"
./build/devinfo > l0-devices.log


# Run L1 tests
echo ""
echo "L1 Tests"
echo "--------"

echo "1/8 Half-duplex (unidirectional) memory copy"
./build/halfduplex > l1-halfduplex.log

echo "2/8 Full-duplex (bidirectional) memory copy"
./build/fullduplex > l1-fullduplex.log

echo "3/8 Half-duplex DMA Read"
./build/uva > l1-uvahalf.log

echo "4/8 Full-duplex DMA Read"
./build/uva --fullduplex > l1-uvafull.log

echo "5/8 Half-duplex DMA Write"
./build/uva --write > l1-uvawhalf.log

echo "6/8 Full-duplex DMA Write"
./build/uva --write --fullduplex > l1-uvawfull.log

echo "7/8 Scatter-Gather"
./build/scatter > l1-scatter.log

echo "8/8 Scaling"
./build/sgemm -n 4096 -k 4096 -m 4096 --repetitions=100 --regression=false --scaling > l1-scaling.log

# Run L2 tests
echo ""
echo "L2 Tests"
echo "--------"

# Matrix multiplication
echo "1/7 Matrix multiplication (correctness)"
./build/sgemm -n 1024 -k 1024 -m 1024 --repetitions=100 --regression=true > l2-sgemm-correctness.log
echo "2/7 Matrix multiplication (performance, single precision)"
./build/sgemm -n 8192 -k 8192 -m 8192 --repetitions=100 --regression=false > l2-sgemm-perf.log
echo "3/7 Matrix multiplication (performance, double precision)"
./build/sgemm -n 2048 -k 2048 -m 2048 --repetitions=100 --regression=false --double > l2-dgemm-perf.log


# Stencil operator
echo "4/7 Stencil (correctness)"
./build/gol --repetitions=5 --regression=true > l2-gol-correctness.log
echo "5/7 Stencil (performance)"
./build/gol --repetitions=1000 --regression=false > l2-gol-perf.log

# Test each GPU separately
echo "6/7 Stencil (single GPU correctness)"
echo "" > l2-gol-single.log
i=0
while [ $i -lt $NUMGPUS ]
do
    echo "GPU $i" >> l2-gol-single.log
    echo "===========" >> l2-gol-single.log
    ./build/gol --num_gpus=1 --repetitions=5 --regression=true --gpuoffset=$i >> l2-gol-single.log
    i=`expr $i + 1`
done


# Temperature test
if [ $TEMPTEST -eq 1 ]
then
    echo "7/7 Cooling"

    # Measure initial temperature
    echo "Initial temp: " > l2-cooling.log
    $NVSMI -q -d TEMPERATURE | grep Current | awk '{print $(NF-1)}' | tr '\n' ' ' >> l2-cooling.log
    echo "" >> l2-cooling.log

    # Wait 5 minutes, measure again
    sleep 300
    echo "Temp after 5min: " >> l2-cooling.log
    $NVSMI -q -d TEMPERATURE | grep Current | awk '{print $(NF-1)}' | tr '\n' ' ' >> l2-cooling.log
    echo "" >> l2-cooling.log

    # Heat, measure temperature right after application
    ./build/sgemm --heat=60 --regression=false --startwith=$NUMGPUS >> l2-cooling.log
    echo "Temp after heat: " >> l2-cooling.log
    $NVSMI -q -d TEMPERATURE | grep Current | awk '{print $(NF-1)}' | tr '\n' ' ' >> l2-cooling.log
    echo "" >> l2-cooling.log

    # Cool for 1 minute, measure again
    sleep 60
    echo "Temp after 1min: " >> l2-cooling.log
    $NVSMI -q -d TEMPERATURE | grep Current | awk '{print $(NF-1)}' | tr '\n' ' ' >> l2-cooling.log
    echo "" >> l2-cooling.log

    # Cool for 4 more minutes, measure again
    sleep 240
    echo "Temp after 5min: " >> l2-cooling.log
    $NVSMI -q -d TEMPERATURE | grep Current | awk '{print $(NF-1)}' | tr '\n' ' ' >> l2-cooling.log
    echo "" >> l2-cooling.log

else
    echo "6/6 Cooling -- SKIPPED"
    echo "SKIPPED" > l2-cooling.log
fi

echo "Done!"
