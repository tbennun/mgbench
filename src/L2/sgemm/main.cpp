// MGBench: Multi-GPU Computing Benchmark Suite
// Copyright (c) 2016, Tal Ben-Nun
// Code adapted from MAPS - Memory Access Pattern Specification Framework
// http://maps-gpu.github.io/
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>

#include <cuda_runtime.h>

#include <gflags/gflags.h>

DEFINE_int32(num_gpus, -1, "Override number of GPUs (or negative to use the amount of available GPUs)");
DEFINE_int32(startwith, 1, "Start with a specific number of GPUs");

bool TestMatMulMAPSMultiUnmodified(int ngpus);

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    bool overall = true;

    int num_actual_gpus = FLAGS_num_gpus;
    if (num_actual_gpus <= 0)
    {
        if (cudaGetDeviceCount(&num_actual_gpus) != cudaSuccess)
        {
            printf("Error %d when getting devices (is CUDA enabled?)\n", num_actual_gpus);
            return 1;
        }
    }
    if (FLAGS_startwith > num_actual_gpus || FLAGS_startwith <= 0)
    {
      printf("Starting with invalid amount of GPUs (Requested: %d, available: %d)\n",
         FLAGS_startwith, num_actual_gpus);
      return 2;
    }

    printf("Running Matrix Multiplication with %d GPUs, starting with %d GPUs\n", num_actual_gpus,
           FLAGS_startwith);

    // Multi-GPU tests
    for (int G = FLAGS_startwith; G <= num_actual_gpus; ++G)
    {
        printf("Testing with %d GPUs\n", G);
        printf("--------------------\n\n");

        overall &= TestMatMulMAPSMultiUnmodified(G);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 3;
    }

    printf("Overall: %s\n", overall ? "OK" : "FAILED");
    
    return overall ? 0 : 4;
}
