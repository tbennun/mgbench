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
DEFINE_uint64(chunksize, 0, "The size of each chunk to transfer (or 0 for one chunk)");
DEFINE_uint64(repetitions, 100, "Number of repetitions to average");
DEFINE_int32(source, -1, "Source to scatter from/gather to (-1 for host)");
DEFINE_bool(ring, false, "Use ring topology for broadcasting");

static void HandleError(const char *file, int line, cudaError_t err)
{
    printf("ERROR in %s:%d: %s (%d)\n", file, line,
           cudaGetErrorString(err), err);
    exit(1);
}

// CUDA assertions
#define CUDA_CHECK(err) do { cudaError_t errr = (err); if(errr != cudaSuccess) { HandleError(__FILE__, __LINE__, errr); } } while(0)

void SetDevice(int dst_dev, int src_dev)
{
   if (dst_dev >= 0)
   {
      CUDA_CHECK(cudaSetDevice(dst_dev));
      return;
   }
   if (src_dev >= 0)
   {
      CUDA_CHECK(cudaSetDevice(src_dev));
   }
}

void *MallocDevice(int dev, size_t size)
{
   void *buff = nullptr;
   if (dev >= 0)
   {
      CUDA_CHECK(cudaSetDevice(dev));
      CUDA_CHECK(cudaMalloc(&buff, size));
   }
   else
   {
      CUDA_CHECK(cudaMallocHost(&buff, size));
   }
   
   return buff;
}

void FreeDevice(int dev, void *buff)
{
   if (dev >= 0)
   {
      CUDA_CHECK(cudaSetDevice(dev));
      CUDA_CHECK(cudaFree(buff));
   }
   else
   {
      CUDA_CHECK(cudaFreeHost(buff));
   }
}


inline void CopyDev2Dev(int dst_dev, void *dst_buff, int src_dev, const void *src_buff,
                        size_t size, cudaStream_t stream)
{
   if (dst_dev < 0) // Device to host
   {
      CUDA_CHECK(cudaMemcpyAsync(dst_buff, src_buff,
                 size, cudaMemcpyDeviceToHost,
                 stream));
   }
   else if (src_dev < 0) // Host to device
   {
      CUDA_CHECK(cudaMemcpyAsync(dst_buff, src_buff,
                 size, cudaMemcpyHostToDevice,
                 stream));
   }
   else // Peer copy
   {
      CUDA_CHECK(cudaMemcpyPeerAsync(dst_buff, dst_dev, src_buff, src_dev,
                     size, stream));
   }
}

double BroadcastRing(int src_dev, int ndevs)
{
    // Setup chunks
    size_t chunk_size = ((FLAGS_chunksize == 0) ? FLAGS_size : FLAGS_chunksize);
    int num_chunks = (FLAGS_size + chunk_size - 1) / chunk_size;
    size_t chunk_remainder = FLAGS_size - (num_chunks - 1) * chunk_size;

    // Setup destination devices
    std::vector<int> dst_devs;
    if (src_dev >= 0)
    {
        // If source device is a GPU, broadcast to all other GPUs only
        for (int i = 1; i < ndevs; ++i)
            dst_devs.push_back((src_dev + i) % ndevs);
    }
    else
    {
        // If source device is the host, broadcast to all GPUs
        for (int i = 0; i < ndevs; ++i)
            dst_devs.push_back(i);
    }

    ndevs = (int)dst_devs.size();
    
    // Setup streams, events and the destination buffers
    std::vector<cudaStream_t> streams (ndevs);
    std::vector<cudaEvent_t> events (ndevs * num_chunks);
    std::vector<char *> buffers (ndevs);
    #define GET_CHUNK(dev, chunk) (events[(dev) * num_chunks + (chunk)])
    
    for (int i = 0; i < ndevs; ++i)
    {
        CUDA_CHECK(cudaSetDevice(dst_devs[i]));

        CUDA_CHECK(cudaMalloc(&buffers[i], FLAGS_size));
        
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i],
                                             cudaStreamNonBlocking));
        for (int c = 0; c < num_chunks; ++c)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&GET_CHUNK(i, c),
                                                cudaEventDisableTiming));
        }
    }

    // Setup source buffer
    void *src_buffer = MallocDevice(src_dev, FLAGS_size);
            
    ////////////////////////////////////////////////

    // Broadcast
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
        size_t curchunk = chunk_size;
        size_t offset = 0;
        
        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            if (chunk == num_chunks - 1)
                curchunk = chunk_remainder;

            /*
              Scheme:
              GPU 1: [  1  ]E[  2  ]E[  3  ]E[  4  ]E
              GPU 2:         [  1  ]E[  2  ]E[  3  ]E[  4  ]E
              GPU 3:                 [  1  ]E[  2  ]E[  3  ]E[  4  ]E
             */
            for (int d = 0; d < ndevs; ++d)
            {
                int src = (d == 0) ? src_dev : dst_devs[d - 1];
                int dst = dst_devs[d];
                const char *src_buff = (const char *)((d == 0) ? src_buffer :
                                                      buffers[d - 1]);

                if (d > 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(streams[d],
                                                   GET_CHUNK(d - 1, chunk), 0));
                }
                CopyDev2Dev(dst, buffers[d] + offset,
                            src, src_buff + offset,
                            curchunk, streams[d]);
                CUDA_CHECK(cudaEventRecord(GET_CHUNK(d, chunk), streams[d]));
            }

            
            offset += curchunk;
        }

        // Sync on last event of last device
        CUDA_CHECK(cudaEventSynchronize(GET_CHUNK(ndevs - 1,
                                                  num_chunks - 1)));
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;

    ////////////////////////////////////////////////
    
    // Teardown

    for (int i = 0; i < ndevs; ++i)
    {
        CUDA_CHECK(cudaSetDevice(dst_devs[i]));

        CUDA_CHECK(cudaFree(buffers[i]));
        
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        for (int c = 0; c < num_chunks; ++c)
        {
            CUDA_CHECK(cudaEventDestroy(GET_CHUNK(i, c)));
        }
    }

    FreeDevice(src_dev, src_buffer);
    
    return mstime;
}


double BroadcastOneToAll(int dst_dev, int src_dev, maps::multi::Barrier *bar)
{
    void *dst_buff = nullptr, *src_buff = nullptr;
    cudaStream_t stream;
    
    // Allocate buffers
    src_buff = MallocDevice(src_dev, FLAGS_size);
    dst_buff = MallocDevice(dst_dev, FLAGS_size);
    
    // Synchronize devices before copying
    SetDevice(dst_dev, src_dev);

    size_t chunk_size = ((FLAGS_chunksize == 0) ? FLAGS_size : FLAGS_chunksize);
    int num_chunks = (FLAGS_size + chunk_size - 1) / chunk_size;
    size_t chunk_remainder = FLAGS_size - (num_chunks - 1) * chunk_size;
   
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaDeviceSynchronize());
    bar->Sync();

  
    // Copy
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
       char *dstp = (char *)dst_buff, *srcp = (char *)src_buff;
       size_t curchunk = chunk_size;
       
       for (int chunk = 0; chunk < num_chunks; ++chunk)
       {
           if (chunk == num_chunks - 1)
               curchunk = chunk_remainder;
           CopyDev2Dev(dst_dev, dstp, src_dev, srcp, curchunk, stream);
           dstp += chunk_size;
           srcp += chunk_size;
       }
       
    }
    SetDevice(dst_dev, src_dev);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    
    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;
    
    // Free buffers
    FreeDevice(src_dev, src_buff);
    FreeDevice(dst_dev, dst_buff);
    
    // Free stream
    SetDevice(dst_dev, src_dev);
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return mstime;
}

void ScatterGatherDeviceThread(int device_id, int src_device,
                               maps::multi::Barrier *bar,
                               std::vector<double> *results)
{
    // Scatter test
    results->at(device_id) = BroadcastOneToAll(device_id, src_device, bar); 
    bar->Sync();

    // Gather test
    results->at(device_id) = BroadcastOneToAll(src_device, device_id, bar);  
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
    if (FLAGS_chunksize > 0)
    {
        printf("Fragment size: %.2f MB\n",
               (FLAGS_chunksize / 1024.0f / 1024.0f));
        printf("Fragments per transfer: %d\n",
               (int)((FLAGS_size + FLAGS_chunksize - 1) / FLAGS_chunksize));
    }
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

    
    // Set up print string
    char source[256] = {0};
    if(FLAGS_source < 0)
        snprintf(source, 256, "Host");
    else
        snprintf(source, 256, "GPU %d", FLAGS_source);
    
    
    if (FLAGS_ring)
    {
        double mstime = BroadcastRing(FLAGS_source, ndevs);

        printf("Broadcast (%s to all GPUs): ", source);
        PrintResult(mstime);
        printf("\n");
        
        return 0;
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
