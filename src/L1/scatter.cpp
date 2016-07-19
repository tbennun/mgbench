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
DEFINE_int32(source, -1, "Source to scatter from/gather to (-1 for host)");

static void HandleError(const char *file, int line, cudaError_t err)
{
    printf("ERROR in %s:%d: %s (%d)\n", file, line,
           cudaGetErrorString(err), err);
    exit(1);
}

// CUDA assertions
#define CUDA_CHECK(err) do { cudaError_t errr = (err); if(errr != cudaSuccess) { HandleError(__FILE__, __LINE__, errr); } } while(0)

double Copy(int dst_dev, int src_dev, maps::multi::Barrier *bar)
{
    void *dst_buff = nullptr, *src_buff = nullptr;
    cudaStream_t stream;
    
    // Allocate buffers
    if (src_dev >= 0)
    {
        CUDA_CHECK(cudaSetDevice(src_dev));
        CUDA_CHECK(cudaMalloc(&src_buff, FLAGS_size));
    }
    else
    {
        CUDA_CHECK(cudaMallocHost(&src_buff, FLAGS_size));
    }
    if (dst_dev >= 0)
    {
        CUDA_CHECK(cudaSetDevice(dst_dev));
        CUDA_CHECK(cudaMalloc(&dst_buff, FLAGS_size));
    }
    else
    {
        CUDA_CHECK(cudaMallocHost(&dst_buff, FLAGS_size));
    }
    
    // Synchronize devices before copying
    int dev = (dst_dev >= 0) ? dst_dev : src_dev;
    CUDA_CHECK(cudaSetDevice(dev));
    
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaDeviceSynchronize());
    bar->Sync();

    // Copy
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
        if (dst_dev < 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(dst_buff, src_buff,
                                       FLAGS_size, cudaMemcpyDeviceToHost,
                                       stream));
        }
        else if (src_dev < 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(dst_buff, src_buff,
                                       FLAGS_size, cudaMemcpyHostToDevice,
                                       stream));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyPeerAsync(dst_buff, dst_dev, src_buff, src_dev,
                                           FLAGS_size, stream));
        }
    }
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;
    
    // Free buffers
    if (src_dev >= 0)
    {
        CUDA_CHECK(cudaSetDevice(src_dev));
        CUDA_CHECK(cudaFree(src_buff));
    }
    else
    {
        CUDA_CHECK(cudaFreeHost(src_buff));
    }
    if (dst_dev >= 0)
    {
        CUDA_CHECK(cudaSetDevice(dst_dev));
        CUDA_CHECK(cudaFree(dst_buff));
    }
    else
    {
        CUDA_CHECK(cudaFreeHost(dst_buff));
    }

    // Free stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    return mstime;
}

void ScatterGatherDeviceThread(int device_id, int src_device,
                               maps::multi::Barrier *bar,
                               std::vector<double> *results)
{
    // Scatter test
    results->at(device_id) = Copy(device_id, src_device, bar); 
    bar->Sync();

    // Gather test
    results->at(device_id) = Copy(src_device, device_id, bar);  
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

    if (FLAGS_source < -1 || FLAGS_source >= ndevs)
    {
        printf("ERROR: Invalid device ID given (%d)\n", FLAGS_source);
        return 1;
    }

    printf("Enabling peer-to-peer access\n");
    
    // Enable peer-to-peer access       
    for(int i = 0; i < ndevs; ++i)
    {
        CUDA_CHECK(cudaSetDevice(i));
        for(int j = 0; j < ndevs; ++j)
            if (i != j)
                cudaDeviceEnablePeerAccess(j, 0);
    } 

    
    std::vector<std::thread> threads;
    maps::multi::Barrier bar (ndevs + 1);
    std::vector<double> results (ndevs, 0.0);
    double avg_result = 0.0, min_result = 0.0;

    // Create threads
    for (int i = 0; i < ndevs; ++i)
        threads.push_back(std::thread(ScatterGatherDeviceThread, i,
                                      FLAGS_source,
                                      &bar, &results));

    // Set up print string
    char source[256] = {0};
    if(FLAGS_source < 0)
        snprintf(source, 256, "Host");
    else
        snprintf(source, 256, "GPU %d", FLAGS_source);
    
    // Scatter test
    printf("Scatter (%s to all GPUs): ", source);
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
    printf("Gather (all GPUs to %s): ", source);
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
