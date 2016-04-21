// MGBench: Multi-GPU Computing Benchmark Suite
// Copyright (c) 2016, Tal Ben-Nun
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
#include <cstdlib>
#include <chrono>
#include <thread>

#include <gflags/gflags.h>

#include <cuda_runtime.h>

#include <maps/multi/worker.h> // For barrier

DEFINE_uint64(size, 100*1024*1024, "The amount of data to transfer");
DEFINE_uint64(repetitions, 100, "Number of repetitions to average");

static void HandleError(const char *file, int line, cudaError_t err)
{
    printf("ERROR in %s:%d: %s (%d)\n", file, line,
           cudaGetErrorString(err), err);
    exit(1);
}

// CUDA assertions
#define CUDA_CHECK(err) do { cudaError_t errr = (err); if(errr != cudaSuccess) { HandleError(__FILE__, __LINE__, errr); } } while(0)

double CopyHostDevice(int dev, bool d2h, maps::multi::Barrier *bar)
{
    void *dev_buff = nullptr, *host_buff = nullptr;

    // Allocate buffers
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMalloc(&dev_buff, FLAGS_size));    
    CUDA_CHECK(cudaMallocHost(&host_buff, FLAGS_size));

    // Synchronize devices before copying
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaDeviceSynchronize());
    bar->Sync();

    // Copy
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
        if (d2h)
        {
            CUDA_CHECK(cudaMemcpyAsync(host_buff, dev_buff,
                                       FLAGS_size, cudaMemcpyDeviceToHost));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(dev_buff, host_buff,
                                       FLAGS_size, cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;
    
    // Free buffers
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(dev_buff));
    CUDA_CHECK(cudaFreeHost(host_buff));

    return mstime;
}

void ScatterGatherDeviceThread(int device_id, maps::multi::Barrier *bar,
                               std::vector<double> *results)
{
    results->at(device_id) = CopyHostDevice(device_id, false, bar); // Scatter test
    bar->Sync();
    
    results->at(device_id) = CopyHostDevice(device_id, true, bar);  // Gather test
    bar->Sync();
}

void AvgMin(const std::vector<double>& vec, double& avg, double& minval)
{
    avg = vec[0];
    minval = vec[0];
    for (size_t i = 1; i < vec.size(); ++i)
    {
        if (vec[i] < minval)
            minval = vec[i];
        avg += vec[i];
    }

    avg /= (double)vec.size();
}

void PrintResult(double mstime)
{
    // MiB/s = [bytes / (1024^2)] / [ms / 1000]
    double MBps = (FLAGS_size / 1024.0 / 1024.0) / (mstime / 1000.0);
    
    printf("%.2lf MB/s (%lf ms)", MBps, mstime);
}

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    printf("Host-GPU memory scatter-gather test\n");
    
    int ndevs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndevs));

    printf("GPUs: %d\n", ndevs);
    printf("Data size: %.2f MB\n", (FLAGS_size / 1024.0f / 1024.0f));
    printf("Repetitions: %d\n", (int)FLAGS_repetitions);
    printf("\n");

    if (ndevs == 0)
        return 0;

    std::vector<std::thread> threads;
    maps::multi::Barrier bar (ndevs + 1);
    std::vector<double> results (ndevs, 0.0);
    double avg_result = 0.0, min_result = 0.0;

    // Create threads
    for (int i = 0; i < ndevs; ++i)
        threads.push_back(std::thread(ScatterGatherDeviceThread, i, &bar, &results));
    
    // Scatter test
    printf("Scatter (host to all GPUs): ");
    bar.Sync();
    bar.Sync();
    AvgMin(results, avg_result, min_result);
    printf("Mean "); PrintResult(avg_result);
    printf(", Max "); PrintResult(min_result);
    printf("\nRaw: %lf", results[0]);
    for (int i = 1; i < ndevs; ++i)
        printf(", %lf", results[i]);
    printf("\n");

    // Gather test
    printf("Gather (all GPUs to host): ");
    bar.Sync();
    bar.Sync();
    AvgMin(results, avg_result, min_result);
    printf("Mean "); PrintResult(avg_result);
    printf(", Max "); PrintResult(min_result);
    printf("\nRaw: %lf", results[0]);
    for (int i = 1; i < ndevs; ++i)
        printf(", %lf", results[i]);
    printf("\n");
    
    // Destroy threads
    for (int i = 0; i < ndevs; ++i)
        threads[i].join();
    
    return 0;
}
