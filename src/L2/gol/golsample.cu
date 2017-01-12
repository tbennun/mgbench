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
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>

#include <vector>
#include <map>
#include <memory>

#include <gflags/gflags.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <maps/maps.cuh>
#include <maps/multi/multi.cuh>

#define BW 16
#define BH 32

DEFINE_int32(width,  16384, "Image width");
DEFINE_int32(height, 16384, "Image height");

DEFINE_bool(multithreading, true, "Run a thread per device");
DEFINE_bool(regression,     true, "Perform regression tests");
DEFINE_int32(repetitions, 100, "Number of iterations for test");
DEFINE_int32(random_seed, -1,  "Override random seed (default is current time)");
unsigned int curtime = (unsigned int)time(NULL);

DEFINE_int32(gpuoffset, 0, "Offset the first used GPU ID");
DEFINE_bool(save_images, true, "Save images if regression test failed");

DEFINE_int32(elemsize, 8, "Element size in bits (options: 8-bit [uint8_t], 16-bit [uint16_t], 32-bit [float], 64-bit [double])");

template<typename T>
void GameOfLife_CPUTick(const T *in, size_t inStride, T *out, size_t outStride, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            T numLiveNeighbors = T(0);
            T isLive;

            for (int k = -1; k <= 1; k++)
            {
                for (int m = -1; m <= 1; m++)
                {
                    T val = in[inStride * maps::Wrap((i + k), height) + maps::Wrap((j + m), width)];
                    if (k == 0 && m == 0)
                        isLive = val;
                    else
                        numLiveNeighbors += val;
                }
            }

            // Game of Life conditions
            if (isLive)
            {
                if (numLiveNeighbors < 2 || numLiveNeighbors > 3)
                    isLive = 0;
            }
            else
            {
                if (numLiveNeighbors == 3)
                    isLive = 1;
            }

            // Fill output cell
            out[i * outStride + j] = isLive;
        }
    }
}

template<typename T, int BLOCK_WIDTH, int BLOCK_HEIGHT, int ITEMS_PER_THREAD, int ROWS_PER_THREAD>
__global__ void GameOfLifeTickMMAPS MAPS_MULTIDEF(maps::Window2D<T, BLOCK_WIDTH, BLOCK_HEIGHT, 
                                                                 1, maps::WB_WRAP, ITEMS_PER_THREAD, ROWS_PER_THREAD> inFrame,
                                                  maps::StructuredInjectiveOutput<T, 2, BLOCK_WIDTH, BLOCK_HEIGHT, 1, 
                                                                                  ITEMS_PER_THREAD, ROWS_PER_THREAD> outFrame)
{
    MAPS_MULTI_INITVARS(inFrame, outFrame);
    
    // If there are no items to write, return
    if (outFrame.Items() == 0)
        return;

    #pragma unroll
    MAPS_FOREACH(oiter, outFrame)
    {
        T numLiveNeighbors = T(0);
        T isLive;

        // Determine number of live neighbors
        /////////////////////////////////////////
        #pragma unroll
        MAPS_FOREACH_ALIGNED(iter, inFrame, oiter)
        {
            if (iter.index() == 4)
                isLive = *iter;
            else
                numLiveNeighbors += *iter;
        }    
        /////////////////////////////////////////

        // Game of Life conditions
        if (isLive)
        {
            if (numLiveNeighbors < 2 || numLiveNeighbors > 3)
                isLive = 0;
        }
        else
        {
            if (numLiveNeighbors == 3)
                isLive = 1;
        }

        // Fill output cell
        *oiter = isLive;
    }

    outFrame.commit();
}

template<typename T>
void SavePGMFile(const T *data, size_t width, size_t height, const char *filename)
{
    if (!FLAGS_save_images)
        return;
    
	FILE *fp = fopen(filename, "wb");
	if (fp)
	{
		fprintf(fp, "P5\n%lu %lu\n1\n", (unsigned long)width, (unsigned long)height);
		fwrite(data, sizeof(T), width * height, fp);
		fclose(fp);
	}
}

template<typename T>
bool GoLCPURegression(const T *otherResult)
{
    if (!FLAGS_regression)
        return true;

    T *dev_inImage = NULL, *dev_outImage = NULL;

    size_t width = FLAGS_width, height = FLAGS_height, inStride = 0, outStride = 0;

    printf("Comparing with CPU...\n");

    srand((FLAGS_random_seed < 0) ? curtime : FLAGS_random_seed);

    // Create input data
    std::vector<T> host_image(width * height, 0);
    for (size_t i = 0; i < width * height; ++i)
        host_image[i] = (rand() < (RAND_MAX / 4)) ? T(1) : T(0);
    
    std::vector<T> host_resultMAPS(width * height, 0);

    inStride = outStride = sizeof(T) * width;

    dev_inImage = &host_image[0];
    dev_outImage = &host_resultMAPS[0];
    
    for (int i = 0; i < FLAGS_repetitions; i++)
    {
        GameOfLife_CPUTick(dev_inImage, inStride / sizeof(T),
                           dev_outImage, outStride / sizeof(T), (int)width, (int)height);
        
        std::swap(dev_inImage, dev_outImage);
    }

    int numErrors = 0;
    for (size_t i = 0; i < width * height; ++i)
    {
        if (dev_inImage[i] != otherResult[i])
        {
#ifdef _DEBUG
            if (FLAGS_print_values)
                printf("ERROR AT INDEX %d, %d: real: %d, other: %d\n", i % width, i / width, 
                       (int)dev_inImage[i], (int)otherResult[i]);
#endif
            numErrors++;
        }
    }
    
    printf("Comparison %s: Errors: %d\n\n", (numErrors == 0) ? "OK" : "FAILED", numErrors);

    if (numErrors > 0)
    {
      SavePGMFile(dev_inImage, width, height, "original.pgm");
      SavePGMFile(otherResult, width, height, "compared.pgm");
    }
    
    return (numErrors == 0);
}

template<typename T, int IPX, int IPY>
bool RunGoL(int ngpus)
{
    size_t width = FLAGS_width, height = FLAGS_height;

    srand((FLAGS_random_seed < 0) ? curtime : FLAGS_random_seed);

    // Create input data
    std::vector<T> host_image(width * height, 0);
    for (size_t i = 0; i < width * height; ++i)
        host_image[i] = (rand() < (RAND_MAX / 4)) ? T(1) : T(0);

    // Create GPU list
    int num_gpus;
    MAPS_CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    std::vector<unsigned int> gpuids;
    for (int i = 0; i < ngpus; ++i)
        gpuids.push_back((i + FLAGS_gpuoffset) % num_gpus);

    // Create scheduler
    maps::multi::Scheduler sched (gpuids);

    if (!FLAGS_multithreading) {
        sched.DisableMultiThreading();
    }
    
    // Define data structures to be used
    maps::multi::Matrix<T> A (width, height), B (width, height);

    A.Bind(&host_image[0]); // Automatic deduction of stride

    std::vector<T> MAPSMulti_result(width * height, 0);
    
    dim3 block_dims(BW, BH, 1);
    dim3 grid_dims(maps::RoundUp((unsigned int)width, block_dims.x), maps::RoundUp((unsigned int)height, block_dims.y), 1);

    // Analyze the memory access patterns for allocation purposes
    maps::multi::AnalyzeCall(sched, grid_dims, block_dims, maps::multi::Window2D<T, BW, BH, 1, maps::WB_WRAP, IPX, IPY>(A),
                             maps::multi::StructuredInjectiveMatrixO<T, IPX, IPY>(B));
    maps::multi::AnalyzeCall(sched, grid_dims, block_dims, maps::multi::Window2D<T, BW, BH, 1, maps::WB_WRAP, IPX, IPY>(B),
                             maps::multi::StructuredInjectiveMatrixO<T, IPX, IPY>(A));


    for (int i = 0; i < num_gpus; i++)
    {
        MAPS_CUDA_CHECK(cudaSetDevice(i));
        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    MAPS_CUDA_CHECK(cudaSetDevice(0));
    auto t1 = std::chrono::high_resolution_clock::now();

    // Invoke the kernels (data exchanges are performed implicitly)
    for (int i = 0; i < FLAGS_repetitions; ++i)
    {
        if (i % 2 == 0)
            maps::multi::Invoke(sched, GameOfLifeTickMMAPS<T, BW, BH, IPX, IPY>, grid_dims, block_dims,
                                maps::multi::Window2D<T, BW, BH, 1, maps::WB_WRAP, IPX, IPY>(A), 
                                maps::multi::StructuredInjectiveMatrixO<T, IPX, IPY>(B));
        else
            maps::multi::Invoke(sched, GameOfLifeTickMMAPS<T, BW, BH, IPX, IPY>, grid_dims, block_dims,
                                maps::multi::Window2D<T, BW, BH, 1, maps::WB_WRAP, IPX, IPY>(B), 
                                maps::multi::StructuredInjectiveMatrixO<T, IPX, IPY>(A));
    }

    sched.WaitAll();
    for (int i = 0; i < num_gpus; i++)
    {
        MAPS_CUDA_CHECK(cudaSetDevice(i));
        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    printf("GoL - MAPS: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_repetitions);

    (FLAGS_repetitions % 2 ? B : A).Bind(&MAPSMulti_result[0]);
    maps::multi::Gather(sched, (FLAGS_repetitions % 2 ? B : A));

    printf("MAPS-Multi: Done!\n");

    return GoLCPURegression(&MAPSMulti_result[0]);
}

bool TestGoLMAPSMulti(int ngpus)
{
    switch (FLAGS_elemsize) {
    default:
        printf("ERROR: Unsupported element size of %d bits given\n", FLAGS_elemsize);
        return false;

    case 8:
        return RunGoL<unsigned char, 4, 2>(ngpus);
    case 16:
        return RunGoL<unsigned short, 2, 2>(ngpus);
    case 32:
        return RunGoL<float, 1, 1>(ngpus);
    case 64:
        return RunGoL<double, 1, 1>(ngpus);
    }
}
