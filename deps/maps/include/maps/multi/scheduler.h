// MAPS - Memory Access Pattern Specification Framework
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

#ifndef __MAPS_MULTI_SCHEDULER_H
#define __MAPS_MULTI_SCHEDULER_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>

#include "../internal/cuda_utils.hpp"
#include "common.h"
#include "memory_analyzer.h"
#include "allocator.h"
#include "task_constructor.h"

#include "worker.h"

namespace maps {
    namespace multi {
        class Scheduler
        {
        protected:

            /**
            * @brief Represents a single device invocation
            *        (including memory copies, kernel/routine launching and event recording)
            */
            class Invocation : public IWork
            {
            public:
                Scheduler& m_scheduler;

            public:
                // For usual kernels
                std::vector<IDatum *> m_kernel_data;
                std::vector<void *> m_kernel_parameters;

                // For unmodified kernels
                std::vector<DatumSegment> m_container_segments;
                std::vector<DatumSegment> m_container_allocation;

                std::vector< std::vector<SegmentCopy> > m_deviceSegmentCopies;

                unsigned int m_numGPUs;
                unsigned int m_gpuId;
                bool m_bUnmodified;

                std::shared_ptr<Task> m_task;
                routine_t m_routine;
                void *m_copiedContext;
                std::shared_ptr< std::vector<uint8_t> > m_context_buffer;

            public:
                explicit Invocation(Scheduler& scheduler)
                    : m_scheduler(scheduler)
                {
                }

                /// operator () for Functor behavior
                virtual void operator()(Barrier *barrier) override
                {
                    CopySegments();
                    Launch();

                    barrier->Sync();
                    RecordEvent();
                    barrier->Sync();
                }

                void CopySegments()
                {
                    for (unsigned int i = 0; i < m_deviceSegmentCopies.size(); ++i)
                    {
                        auto& copyList = m_deviceSegmentCopies[i];
                        for (auto& copy : copyList)
                        {
                            m_scheduler.CopySegment(i, m_gpuId,
                                                    std::get<0>(copy), std::get<1>(copy), std::get<2>(copy),
                                                    m_scheduler.m_streams[m_gpuId]);
                        }
                    }
                }

                bool Launch()
                {
                    if (!m_task)
                        return true;

                    if (m_bUnmodified)
                    {
                        if (!m_scheduler.CallUnmodifiedRoutine(m_routine, m_copiedContext, *m_context_buffer,
                                                               m_gpuId, m_task->segmentation[m_gpuId],
                                                               m_kernel_parameters, m_container_segments,
                                                               m_kernel_data, m_container_allocation))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        if (!m_scheduler.LaunchKernel(m_task->kernel, m_gpuId, m_task->segmentation[m_gpuId],
                                                      m_kernel_parameters, m_task->dsmem, m_container_segments,
                                                      m_kernel_data))
                        {
                            return false;
                        }
                    }

                    return true;
                }

                void RecordEvent()
                {
                    if (!m_task)
                        return;
                    m_scheduler.RecordEvent(m_gpuId);
                }
            };

            class CopyFromHostAction : public IWork
            {
            private:
                bool m_bFill;

                IDatum *m_datum;
                unsigned int m_gpuid;

                void * m_dst;
                size_t m_dpitch;

                int m_fillValue;
                size_t m_width;
                size_t m_height;

                const void *m_src;
                size_t m_spitch;

                cudaStream_t m_stream;

            public:

                CopyFromHostAction(bool bFill, IDatum *datum, unsigned int gpuid, void *dst, size_t dpitch, int fillValue,
                                   size_t width, size_t height, const void *src, size_t spitch, cudaStream_t stream)
                    : m_bFill(bFill), m_datum(datum), m_gpuid(gpuid), m_dst(dst), m_dpitch(dpitch), m_fillValue(fillValue),
                    m_width(width), m_height(height), m_src(src), m_spitch(spitch), m_stream(stream)
                {
                }

                /// operator () for Functor behavior
                virtual void operator()(Barrier *barrier) override
                {
                  if (m_bFill)
                    MAPS_CUDA_CHECK(cudaMemset2DAsync(m_dst, m_dpitch, m_fillValue,
                                                      m_width, m_height,
                                                      m_stream));

                  else // Otherwise, use memcpy
                    MAPS_CUDA_CHECK(cudaMemcpy2DAsync(m_dst, m_dpitch, m_src, m_spitch,
                                                      m_width, m_height, cudaMemcpyHostToDevice,
                                                      m_stream));                
                }
            };

            /**
            * @brief A single threaded Invoker
            */
            class Invoker : public Worker<>
            {
            private:
                Scheduler& m_scheduler;
                const unsigned int m_gpuId;

            protected:
                virtual void OnStart() override
                {
                    m_scheduler.SetDevice(m_gpuId);
                    // TODO(later): Should we set core affinity explicitly?
                }

            public:
                Invoker(std::shared_ptr<Barrier> barrier, Scheduler& scheduler, unsigned int gpuId)
                    : Worker<>(barrier), m_scheduler(scheduler), m_gpuId(gpuId)
                {
                    this->Run();
                }
            };

        protected:
            /// The GPUs to use for multi-GPU scheduling
            std::vector<unsigned int> m_activeGPUs;

            /// Task counter for manager. Task 0 functions as the "null task".
            unsigned int m_taskCounter;

            /// Tasks are evicted from map when they are done executing (e.g., after gather/wait)
            std::map< taskHandle_t, std::shared_ptr<Task> > m_tasks;

            /// Memory allocation analyzer for data structures
            MemoryAnalyzer m_analyzer;

            /// Memory allocator
            std::shared_ptr<IAllocator> m_allocator;

            /// Used for scheduling copies between tasks (last output location of segments)
            std::map<IDatum *, DatumLocation> m_lastLocation;

            /// Used to conserve memory copies if memory hasn't changed
            std::vector< std::map<IDatum *, std::vector<DatumSegment> > > m_upToDateLocations;

            /// Aggregator manager
            std::map<IDatum *, std::shared_ptr<IAggregator> > m_aggregators;

            /// Buffer manager
            BufferList m_buffers;

            /// MAPS structure managers
            std::set< std::shared_ptr<::maps::IInputContainer> > m_inputContainers;
            std::set< std::shared_ptr<::maps::IOutputContainer> > m_outputContainers;

            std::set< std::shared_ptr<::maps::IContainerComposition> > m_composedContainers;

            /// Graph buffer manager
            std::set<IDatum *> m_dataFromGraphs;

            /// Event manager. Kept for cleanup
            std::vector<cudaEvent_t> m_events;

            /// Stream manager. Kept for cleanup
            std::vector<cudaStream_t> m_streams;

            /// Close-to-optimal exchange order between active GPUs
            /// @note Currently in use only by the TaskGraph since the scheduler is now multi-threaded
            /// see the EnqueueInvocations virtual method
            std::vector<std::pair<unsigned int, unsigned int> > m_exchangeOrder;

            /// Workers for multithreading the actual invoke work
            std::vector< std::unique_ptr<Invoker> > m_workers;

            /// Flag for multi-threading per device
            bool m_bMultiThreaded;

        protected:
            void SetActiveDevices(const std::vector<unsigned int>& devices)
            {
                // Scheduler is multi-threaded by default
                m_bMultiThreaded = true;

                // Get actual number of GPUs
                int count = 0;
                MAPS_CUDA_CHECK(cudaGetDeviceCount(&count));

                // Fill with chosen device IDs
                m_activeGPUs.clear();
                m_activeGPUs.reserve(devices.size());
                if (devices.empty())
                {
                    for (int i = 0; i < count; ++i)
                    {
                        m_activeGPUs.push_back(static_cast<unsigned int>(i));
                    }
                }
                else
                {
                    for (unsigned int dev : devices)
                    {
                        if (dev >= static_cast<unsigned int>(count))
                            printf("Skipping invalid device ID %u\n", dev);
                        else
                        {
                            m_activeGPUs.push_back(dev);
                        }
                    }
                }

                // Enable peer-to-peer access
                int dev = 0;
                MAPS_CUDA_CHECK(cudaGetDevice(&dev));
                for (size_t i = 0; i < m_activeGPUs.size(); ++i)
                {
                    MAPS_CUDA_CHECK(cudaSetDevice(m_activeGPUs[i]));
                    for (size_t j = 0; j < m_activeGPUs.size(); ++j)
                        if (m_activeGPUs[i] != m_activeGPUs[j])
                            cudaDeviceEnablePeerAccess(m_activeGPUs[j], 0);
                }
                MAPS_CUDA_CHECK(cudaSetDevice(dev));

                // Reset segments and other scheduler state fields
                m_analyzer.Reset(m_activeGPUs.size());

                m_buffers.clear();
                m_tasks.clear();
                m_lastLocation.clear();
                m_upToDateLocations.clear();
                m_dataFromGraphs.clear();
                m_aggregators.clear();

                m_buffers.resize(m_activeGPUs.size());
                m_upToDateLocations.resize(m_activeGPUs.size());

                m_events.clear();
                m_streams.clear();
                m_exchangeOrder.clear();

                // Destruct all workers. Actually joins on all.
                m_workers.clear();

                CreateStreams();
            }

            void CreateStreams()
            {
                m_streams.resize(m_activeGPUs.size());
                m_events.resize(m_activeGPUs.size());

                int lastDev = 0;
                cudaGetDevice(&lastDev);

                int i = 0;
                for (unsigned int dev : m_activeGPUs)
                {
                    MAPS_CUDA_CHECK(cudaSetDevice(dev));
                    MAPS_CUDA_CHECK(cudaStreamCreateWithFlags(&m_streams[i], cudaStreamNonBlocking));
                    MAPS_CUDA_CHECK(cudaEventCreateWithFlags(&m_events[i], cudaEventDisableTiming));

                    ++i;
                }

                // Prepare single-thread exchange order (up to 4 GPUs)
                // TODO(later): Extend to N GPUs

                // Adjacent GPUs first (different bus)
                for (unsigned int j = 0; j < m_activeGPUs.size(); j += 2)
                {
                    if ((j + 1) < m_activeGPUs.size())
                    {
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j, j + 1));
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j + 1, j));
                    }
                }
                // Then all the rest
                for (unsigned int j = 0; j < m_activeGPUs.size(); ++j)
                {
                    if ((j + 2) < m_activeGPUs.size())
                    {
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j, j + 2));
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j + 2, j));
                    }
                }
                for (unsigned int j = 1; j < m_activeGPUs.size(); j += 2)
                {
                    if ((j + 1) < m_activeGPUs.size())
                    {
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j, j + 1));
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j + 1, j));
                    }
                }
                for (unsigned int j = 0; j < m_activeGPUs.size(); j += 3)
                {
                    if ((j + 3) < m_activeGPUs.size())
                    {
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j, j + 3));
                        m_exchangeOrder.push_back(std::pair<unsigned int, unsigned int>(j + 3, j));
                    }
                }

                // Restore CUDA context state
                MAPS_CUDA_CHECK(cudaSetDevice(lastDev));
            }

        public:
            Scheduler() : m_taskCounter(0), m_allocator(new SimpleAllocator())
            {
                // Use all devices by default
                SetActiveDevices({});
            }

            Scheduler(const std::vector<unsigned int>& devices)
                : m_taskCounter(0), m_allocator(new SimpleAllocator())
            {
                SetActiveDevices(devices);
            }

            Scheduler(const std::initializer_list<unsigned int>& devices)
                : m_taskCounter(0), m_allocator(new SimpleAllocator())
            {
                SetActiveDevices(devices);
            }

            void SetAllocator(std::shared_ptr<IAllocator>& allocator)
            {
                m_allocator = allocator;
            }

            std::vector<unsigned int> GetActiveDevices() const
            {
                return m_activeGPUs;
            }

            virtual ~Scheduler()
            {
                // Destruct all workers. Actually joins on all
                m_workers.clear();

                // Wait for all computations to finish
                WaitAll();

                // Free events
                for (cudaEvent_t& event : m_events)
                    cudaEventDestroy(event);

                // Free streams
                for (cudaStream_t& stream : m_streams)
                    cudaStreamDestroy(stream);

                // Allocator will free data automatically on destruction
            }

            template<typename... Args>
            void AnalyzeCall(dim3 grid_dims, dim3 block_dims, const Args&... args)
            {
                // TODO(later): If a task invocation or fill command have been invoked, do not allow further analysis
                
                // This function acts as the first pass of the scheduler on the tasks
                Task task;

                // Construct task
                ConstructArgs(task, args...);
                task.grid_size = grid_dims;
                task.block_size = block_dims;

                // Construct task grid segmentation
                ::maps::multi::ConstructSegmentation(task, (unsigned int)m_activeGPUs.size(), false);

                // Analyze the task for memory allocation
                m_analyzer.AnalyzeTask(task);
            }

            template<typename... Args>
            void AnalyzeCallAll(dim3 grid_dims, dim3 block_dims, const Args&... args)
            {
                // This function acts as the first pass of the scheduler on the tasks
                Task task;

                // Construct task
                ConstructArgs(task, args...);
                task.grid_size = grid_dims;
                task.block_size = block_dims;

                // Construct task grid segmentation                
                ::maps::multi::ConstructSegmentation(task, (unsigned int)m_activeGPUs.size(), true);

                // Analyze the task for memory allocation
                m_analyzer.AnalyzeTask(task);
            }

            template<typename D>
            bool Fill(D& actual_datum, int value = 0)
            {
                IDatum *datum = dynamic_cast<IDatum *>(&actual_datum);
                if (datum == nullptr)
                {
                    printf("ERROR: Requested to fill a non-datum data structure\n");
                    return false;
                }
                return Fill(datum, value);
            }

            bool Fill(IDatum *datum, int value = 0)
            {
                if (datum == nullptr)
                {
                    printf("ERROR: Requested to fill a non-datum data structure\n");
                    return false;
                }

                unsigned int numGPUs = (unsigned int)m_activeGPUs.size();

                
                DatumLocation loc;
                loc.state = LS_SEGMENTED;
                

                DatumSegment allocated_segment;
                for (unsigned int i = 0; i < numGPUs; ++i)
                {
                    SetDevice(i);

                    // Get the allocated segment
                    if (!m_analyzer.GetSegment(i, datum, allocated_segment))
                    {
                        printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                        return false;
                    }

                    if (m_buffers[i].find(datum) == m_buffers[i].end())
                    {
                        Memory mem;
                        mem.ptr = AllocateBuffer(i, datum, allocated_segment, datum->GetElementSize(), mem.stride_bytes);
#ifndef NDEBUG
                        if (mem.ptr == nullptr)
                        {
                            printf("ERROR allocating filled buffer on GPU %d\n", i);
                            return false;
                        }
#endif
                        m_buffers[i][datum] = mem;
                    }

                    DatumSegment seg = allocated_segment;
                    seg.m_bFill = true;
                    seg.m_fillValue = value;

                    CopyFromHost(i, datum, allocated_segment, seg, m_streams[i]);

                    loc.entries.push_back(std::make_pair(i, allocated_segment));                    
                }

                // Override last location
                m_lastLocation[datum] = loc;
                for (unsigned int i = 0; i < m_activeGPUs.size(); ++i)
                {
                    if (m_upToDateLocations[i].find(datum) != m_upToDateLocations[i].end())
                        m_upToDateLocations[i][datum].clear();
                    else
                        m_upToDateLocations[i][datum] = std::vector<DatumSegment>();

                    m_upToDateLocations[i][datum].push_back(loc.entries[i].second);
                }

                return true;
            }

            virtual bool CopyFromHost(int dstDevice, IDatum *datum, const DatumSegment& allocated_seg,
                                      const DatumSegment& seg, cudaStream_t stream)
            {
#ifndef NDEBUG
                if (!datum)
                    return false;
#endif
                void *src_ptr = datum->HostPtrMutable();
#ifndef NDEBUG
                if (dstDevice >= m_activeGPUs.size())
                {
                    printf("ERROR: Invalid device ID\n");
                    return false;
                }
                if (seg.GetDimensions() != datum->GetDataDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
                if (m_buffers[dstDevice].find(datum) == m_buffers[dstDevice].end())
                {
                    printf("ERROR: H2D - Memory not allocated on device\n");
                    return false;
                }
                if (!seg.m_bFill && src_ptr == nullptr)
                {
                    printf("ERROR: H2D - Memory not registered on host (2)\n");
                    return false;
                }
#endif                
                Memory mem_dst = m_buffers[dstDevice][datum];
                void *dst_ptr = mem_dst.ptr;

                // Compute real offset
                src_ptr = seg.OffsetPtrBounds(src_ptr, datum, datum->GetElementSize(), datum->GetHostStrideBytes());
                dst_ptr = allocated_seg.OffsetPtr(seg.m_offset, dst_ptr, datum->GetElementSize(), mem_dst.stride_bytes);

                size_t otherdims = 1;
                for (unsigned int i = 1; i < seg.GetDimensions(); ++i)
                    otherdims *= seg.GetDimension(i);

                if (m_bMultiThreaded)
                {
                    LazyInitWorkers();
                    CopyFromHostAction* action = new CopyFromHostAction(seg.m_bFill, datum, dstDevice, dst_ptr, mem_dst.stride_bytes,
                                                                        seg.m_fillValue, datum->GetElementSize() * seg.GetDimension(0),
                                                                        otherdims, src_ptr, datum->GetHostStrideBytes(), stream);

                    m_workers[dstDevice]->Enqueue(std::shared_ptr<IWork>(action));
                }
                else
                {
                    // We need to SetDevice only for the main thread
                    MAPS_CUDA_CHECK(cudaSetDevice(m_activeGPUs[dstDevice]));

                    // If segment requires us to set memory, use memset
                    if (seg.m_bFill)
                      MAPS_CUDA_CHECK(cudaMemset2DAsync(dst_ptr, mem_dst.stride_bytes, seg.m_fillValue,
                                                        datum->GetElementSize() * seg.GetDimension(0), otherdims,
                                                        stream));
                    else // Otherwise, use memcpy
                      MAPS_CUDA_CHECK(cudaMemcpy2DAsync(dst_ptr, mem_dst.stride_bytes, src_ptr, datum->GetHostStrideBytes(),
                                                        datum->GetElementSize() * seg.GetDimension(0), otherdims, cudaMemcpyHostToDevice,
                                                        stream));
                }

                return true;
            }

            virtual bool CopyToHost(int srcDevice, IDatum *datum, const DatumSegment& allocated_seg,
                                    const DatumSegment& seg, cudaStream_t stream, bool async = true, const DatumSegment& host_allocated_seg = DatumSegment())
            {
                bool result = true;

#ifndef NDEBUG
                if (!datum)
                    return false;
#endif
                void *dst_ptr = datum->HostPtrMutable();
#ifndef NDEBUG
                if (srcDevice >= m_activeGPUs.size())
                {
                    printf("ERROR: Invalid device ID\n");
                    return false;
                }
                if (seg.GetDimensions() != datum->GetDataDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
                if (m_buffers[srcDevice].find(datum) == m_buffers[srcDevice].end())
                {
                    printf("ERROR: D2H - Memory not allocated on device\n");
                    return false;
                }
                if (dst_ptr == nullptr)
                {
                    printf("ERROR: D2H - Memory not registered on host (2)\n");
                    return false;
                }
#endif
                Memory mem_src = m_buffers[srcDevice][datum];
                void *src_ptr = mem_src.ptr;
                if (src_ptr == UNALLOCATED_BUFFER)
                {
                    printf("ERROR: Buffer was not yet allocated. Have you called the TaskGraph yet?\n");
                    return false;
                }

                // Compute real offset                
                src_ptr = allocated_seg.OffsetPtr(seg.m_offset, src_ptr, datum->GetElementSize(), mem_src.stride_bytes);
                if (host_allocated_seg.GetDimensions() > 0)
                    dst_ptr = host_allocated_seg.OffsetPtr(seg.m_offset, dst_ptr, datum->GetElementSize(), datum->GetHostStrideBytes());
                else
                    dst_ptr = seg.OffsetPtrBounds(dst_ptr, datum, datum->GetElementSize(), datum->GetHostStrideBytes());

                size_t otherdims = 1;
                for (unsigned int i = 1; i < seg.GetDimensions(); ++i)
                    otherdims *= seg.GetDimension(i);

                MAPS_CUDA_CHECK(cudaSetDevice(m_activeGPUs[srcDevice]));

                MAPS_CUDA_CHECK(cudaStreamWaitEvent(stream, m_events[srcDevice], 0));

                MAPS_CUDA_CHECK(cudaMemcpy2DAsync(dst_ptr, datum->GetHostStrideBytes(), src_ptr, mem_src.stride_bytes,
                                                  datum->GetElementSize() * seg.GetDimension(0), otherdims, cudaMemcpyDeviceToHost,
                                                  stream));

                if (!async)
                    cudaStreamSynchronize(stream);

                return result;
            }

            virtual bool CopySegment(unsigned int srcDevice, unsigned int dstDevice, IDatum *datum,
                                     const DatumSegment& segment_src, const DatumSegment& segment_dst,
                                     cudaStream_t stream)
            {
#ifndef NDEBUG
                if (srcDevice == dstDevice)
                {
                    printf("ERROR: Cannot copy segment from device to itself\n");
                    return false;
                }
                if (srcDevice >= m_activeGPUs.size() || dstDevice >= m_activeGPUs.size())
                {
                    printf("ERROR: Invalid GPU IDs\n");
                    return false;
                }
#endif
                DatumSegment alloc_seg_src, alloc_seg_dst;
                if (!m_analyzer.GetSegment(srcDevice, datum, alloc_seg_src))
                {
                    printf("ERROR: No segment allocated at source\n");
                    return false;
                }
                if (!m_analyzer.GetSegment(dstDevice, datum, alloc_seg_dst))
                {
                    printf("ERROR: No segment allocated at destination\n");
                    return false;
                }
                unsigned int ndims = segment_dst.GetDimensions();
#ifndef NDEBUG                
                if (ndims != alloc_seg_src.GetDimensions() || ndims != alloc_seg_dst.GetDimensions() ||
                    ndims != segment_src.GetDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
                if (m_buffers[srcDevice].find(datum) == m_buffers[srcDevice].end())
                {
                    printf("ERROR: Memory not allocated on device\n");
                    return false;
                }
                if (m_buffers[dstDevice].find(datum) == m_buffers[dstDevice].end())
                {
                    printf("ERROR: Memory not allocated on device (2)\n");
                    return false;
                }
#endif
                Memory mem_src, mem_dst;
                mem_src = m_buffers[srcDevice][datum];
                mem_dst = m_buffers[dstDevice][datum];
                unsigned char *ptr_src, *ptr_dst;

                ptr_src = (unsigned char *)alloc_seg_src.OffsetPtr(segment_src.m_offset, mem_src.ptr, datum->GetElementSize(),
                                                                   mem_src.stride_bytes);
                ptr_dst = (unsigned char *)alloc_seg_dst.OffsetPtr(segment_dst.m_offset, mem_dst.ptr, datum->GetElementSize(),
                                                                   mem_dst.stride_bytes);

                std::vector<int64_t> src_end_offset = segment_src.m_offset;
                for (unsigned int i = 0; i < ndims; ++i)
                    src_end_offset[i] += (int64_t)segment_src.m_dimensions[i] - 1;

                unsigned char *ptr_src_end = (unsigned char *)alloc_seg_src.OffsetPtr(src_end_offset, mem_src.ptr,
                                                                                      datum->GetElementSize(), mem_src.stride_bytes);

                std::vector<int64_t> dst_end_offset = segment_dst.m_offset;
                for (unsigned int i = 0; i < ndims; ++i)
                    dst_end_offset[i] += (int64_t)segment_dst.m_dimensions[i] - 1;

                unsigned char *ptr_dst_end = (unsigned char *)alloc_seg_dst.OffsetPtr(dst_end_offset,
                                                                                      mem_dst.ptr, datum->GetElementSize(),
                                                                                      mem_dst.stride_bytes);
#ifndef NDEBUG
                ptrdiff_t bytes_src = (ptr_src_end - ptr_src + datum->GetElementSize()), bytes_dst = (ptr_dst_end - ptr_dst + datum->GetElementSize());
                if (bytes_src != bytes_dst)
                {
                    printf("Can only copy contiguous segments at this time\n");
                    return false;
                }
                if (ptr_src_end <= ptr_src || ptr_dst_end <= ptr_dst)
                {
                    printf("ERROR: Invalid segments\n");
                    return false;
                }
#endif                
                size_t bytes = (size_t)(ptr_src_end - ptr_src + datum->GetElementSize());
                bool result = true;

                if (segment_dst.m_bFill)
                {
                    MAPS_CUDA_CHECK(cudaSetDevice(m_activeGPUs[dstDevice]));
                }
                else
                {
                    // Let the stream wait for the event first
                    MAPS_CUDA_CHECK(cudaStreamWaitEvent(stream, m_events[srcDevice], 0));
                }

                if (segment_dst.m_bFill) // If segment requires us to set memory, use memset
                {
                  MAPS_CUDA_CHECK(cudaMemsetAsync(ptr_dst, segment_dst.m_fillValue,
                                                  bytes, stream));
                }
                else
                {
                    // TODO(later): Use memcpy3DPeer?
                    MAPS_CUDA_CHECK(cudaMemcpyPeerAsync(ptr_dst, m_activeGPUs[dstDevice], ptr_src,
                                                        m_activeGPUs[srcDevice],
                                                        bytes, stream));
                }
                
                return result;
            }

            virtual void SetDevice(unsigned int gpuID)
            {
                MAPS_CUDA_CHECK(cudaSetDevice((int)m_activeGPUs[gpuID]));
            }

            virtual void *AllocateBuffer(unsigned int gpuID, IDatum *datum, const DatumSegment& segment, size_t elementSize, size_t& strideBytes)
            {
                return m_allocator->Allocate(m_activeGPUs[gpuID], segment, elementSize, strideBytes);
            }

            virtual bool LaunchKernel(const void *kernel, int deviceIdx, const GridSegment& segmentation,
                                      std::vector<void *>& kernel_parameters, size_t dsmem,
                                      const std::vector<DatumSegment>& /*container_segments*/, std::vector<IDatum *>& /*kernel_data*/)
            {
                cudaError_t err = cudaSuccess;

                uint3 block_offset = make_uint3(segmentation.offset, 0, 0);
                dim3 total_grid_dims = segmentation.total_grid_dims;

                // First, push the MAPS_MULTIDEF arguments
                // (unsigned int deviceIdx, dim3 multigridDim, uint3 blockIdx)
                kernel_parameters[0] = &deviceIdx;
                kernel_parameters[1] = &total_grid_dims;
                kernel_parameters[2] = &block_offset;

                // Launch the kernel
                err = cudaLaunchKernel(kernel, segmentation.blocks, segmentation.block_dims,
                                       &kernel_parameters[0], dsmem, m_streams[deviceIdx]);
                if (err != cudaSuccess)
                {
                    printf("ERROR launching kernel (%d): %s\n", err, cudaGetErrorString(err));
                    return false;
                }

#ifdef _DEBUG
                err = cudaStreamSynchronize(m_streams[deviceIdx]);
                if (err != cudaSuccess)
                {
                    printf("ERROR running kernel on GPU %d (%d): %s\n", deviceIdx + 1, err, cudaGetErrorString(err));
                    return (taskHandle_t)0;
                }
#endif

                return true;
            }

            virtual bool CallUnmodifiedRoutine(routine_t routine, void *context, std::vector<uint8_t>& copied_context, int deviceIdx,
                                               const GridSegment& segmentation,
                                               std::vector<void *>& kernel_parameters,
                                               const std::vector<DatumSegment>& container_segments, std::vector<IDatum *>& /*kernel_data*/,
                                               const std::vector<DatumSegment>& container_allocation)
            {
#ifndef NDEBUG
                if (!routine)
                {
                    printf("ERROR: null routine input\n");
                    return false;
                }
#endif

                return routine(context, deviceIdx, m_streams[deviceIdx], segmentation, kernel_parameters, container_segments, container_allocation);
            }

            virtual bool RecordEvent(int deviceIdx, int laneIdx = 0)
            {
                MAPS_CUDA_CHECK(cudaEventRecord(m_events[deviceIdx], m_streams[deviceIdx]));
                return true;
            }

        protected:
            virtual void EnqueueInvocations(std::vector< std::shared_ptr<Invocation> >& invocations)
            {
                if (m_bMultiThreaded) {
                    RunInvocationsOnWorkers(invocations);
                }

                else {
                    RunInvocationsSequentially(invocations);
                }
            }

            void LazyInitWorkers()
            {
                size_t numGPUs = m_activeGPUs.size();

                // Lazy loading to avoid the creation of threads by the TaskGraph
                if (m_workers.size() == 0)
                {
                    m_workers.resize(numGPUs);
                    std::shared_ptr<Barrier> barrier = std::make_shared<Barrier>(numGPUs);

                    for (size_t i = 0; i < numGPUs; ++i)
                    {
                        m_workers[i] = std::unique_ptr<Invoker>(new Invoker(barrier, *this, (unsigned int)i));
                    }
                }
            }

            void RunInvocationsOnWorkers(std::vector< std::shared_ptr<Invocation> >& invocations)
            {
                size_t numGPUs = m_activeGPUs.size();
                LazyInitWorkers();

                int dev_ctr = 0;
                for (size_t i = 0; i < numGPUs; ++i)
                {
                    m_workers[i]->Enqueue(std::shared_ptr<IWork>(invocations[dev_ctr++]));
                }
            }

            void RunInvocationsSequentially(std::vector< std::shared_ptr<Invocation> >& invocations)
            {
                // This method basically contains the former sequential implementation of multi-device
                // kernel launching

                size_t numGPUs = this->m_activeGPUs.size();

                // Perform all data exchanges prior to kernels
                for (size_t i = 0; i < this->m_exchangeOrder.size(); ++i)
                {
                    auto& exchg_order = this->m_exchangeOrder[i];

                    SetDevice(exchg_order.second);

                    // Disabled device
                    if (invocations[exchg_order.second]->m_task == nullptr)
                        continue;

                    auto& copyList = invocations[exchg_order.second]->m_deviceSegmentCopies[exchg_order.first];
                    for (auto& copy : copyList)
                    {
                        CopySegment(exchg_order.first, exchg_order.second,
                                    std::get<0>(copy), std::get<1>(copy), std::get<2>(copy),
                                    this->m_streams[exchg_order.second]);
                    }
                }

                // Run all kernels                
                for (size_t i = 0; i < numGPUs; ++i)
                {
                    SetDevice((unsigned int)i);
                    invocations[i]->Launch();
                    invocations[i]->RecordEvent();
                }
            }

        public:
            template<typename Kernel, typename... Args>
            inline taskHandle_t Invoke(Kernel kernel, dim3 grid_dims,
                                       dim3 block_dims, const Args&... args)
            {
                return InvokeInternal(kernel, grid_dims, block_dims, 0, false, nullptr, nullptr, 0, false, false, args...);
            }

            template<typename Kernel, typename... Args>
            taskHandle_t InvokeDynamicSMem(Kernel kernel, dim3 grid_dims, dim3 block_dims,
                                           size_t dynamic_smem, const Args&... args)
            {
                return InvokeInternal(kernel, grid_dims, block_dims, dynamic_smem, false, nullptr, nullptr, 0, false, false, args...);
            }

            template<typename... Args>
            taskHandle_t InvokeUnmodified(routine_t routine, void *context, dim3 work_dims,
                                          const Args&... args)
            {
                return InvokeInternal(nullptr, work_dims, dim3(), 0, true, routine, context, 0, false, false, args...);
            }

            template<typename Kernel, typename... Args>
            inline taskHandle_t InvokeAll(Kernel kernel, dim3 grid_dims,
                                          dim3 block_dims, const Args&... args)
            {
                return InvokeInternal(kernel, grid_dims, block_dims, 0, false, nullptr, nullptr, 0, true, false, args...);
            }

            template<typename Kernel, typename... Args>
            taskHandle_t InvokeAllDynamicSMem(Kernel kernel, dim3 grid_dims, dim3 block_dims,
                                              size_t dynamic_smem, const Args&... args)
            {
                return InvokeInternal(kernel, grid_dims, block_dims, dynamic_smem, false, nullptr, nullptr, 0, true, false, args...);
            }

            template<typename... Args>
            taskHandle_t InvokeAllUnmodified(routine_t routine, void *context, dim3 work_dims,
                                             const Args&... args)
            {
                return InvokeInternal(nullptr, work_dims, dim3(), 0, true, routine, context, 0, true, false, args...);
            }

            void DisablePitchedAllocation()
            {
                m_allocator->DisablePitchedAllocation();
            }

            void DisableMultiThreading()
            {
                m_bMultiThreaded = false;
            }

        protected:

            // Hook for each task, used for checkpoint/restart
            virtual void OnBeginTask(Task& task) {}

            template<typename Kernel, typename... Args>
            taskHandle_t InvokeInternal(Kernel kernel, dim3 grid_dims, dim3 block_dims,
                                        size_t dynamic_smem, bool bUnmodified, routine_t routine,
                                        void *context, size_t context_size, bool bInvokeAll, bool bSkip, const Args&... args)
            {
                // This function acts as the first pass of the scheduler on the tasks
                std::shared_ptr<Task> task_ptr = std::make_shared<Task>();
                Task& task = *task_ptr;

                // User-based dynamic SMem is allocated before task parameters
                task.dsmem = dynamic_smem;


                // Construct a temporary task for segmentation purposes
                ConstructCall(task, kernel);
                ConstructArgs(task, args...);
                task.grid_size = grid_dims;
                task.block_size = block_dims;

                // Construct task grid segmentation
                ConstructSegmentation(task, (unsigned int)m_activeGPUs.size(), bInvokeAll);

                return InvokeInternal(task_ptr, bUnmodified, routine, context, context_size, bInvokeAll, bSkip);
            }

            taskHandle_t InvokeInternal(std::shared_ptr<Task>& task_ptr, bool bUnmodified, routine_t routine,
                                        void *context, size_t context_size, bool bInvokeAll, bool bSkip)
            {
                unsigned int total_gpus = (unsigned int)m_activeGPUs.size();
                Task& task = *task_ptr;

                // Construct container segmentation
                for (auto&& input : task.inputs)
                    input.segmenter->Segment(task.segmentation, input.segmentation);
                for (auto&& output : task.outputs)
                    output.segmenter->Segment(task.segmentation, output.segmentation);

                OnBeginTask(task);

                // If should copy context, allocate new one and copy the current context to the newly created one
                void *copiedContext = context;

                std::shared_ptr< std::vector<uint8_t> > context_buffer_ptr = std::make_shared< std::vector<uint8_t> >();
                std::vector<uint8_t>& context_buffer = *context_buffer_ptr;
                if (context_size > 0)
                {
                    context_buffer.resize(context_size);
                    memcpy(&context_buffer[0], context, context_size);
                    copiedContext = &context_buffer[0];
                }

                std::vector< std::shared_ptr<Invocation> > invocations(total_gpus);
                for (unsigned int i = 0; i < total_gpus; i++)
                {
                    invocations[i] = std::make_shared<Invocation>(*this);

                    invocations[i]->m_task = task_ptr;
                    invocations[i]->m_routine = routine;
                    invocations[i]->m_copiedContext = copiedContext;
                    invocations[i]->m_context_buffer = context_buffer_ptr;
                    invocations[i]->m_gpuId = i;
                    invocations[i]->m_numGPUs = total_gpus;
                    invocations[i]->m_bUnmodified = bUnmodified;
                }

                // Create kernel parameters
                size_t nargs = task.argument_ordering.size();

                std::map<IDatum *, DatumLocation> locationUpdates;

                // Prepare for per-GPU kernel launch
                for (unsigned int i = 0; i < total_gpus; ++i)
                {
                    int arg_offset = (bUnmodified ? 0 : 3);

                    std::vector<IDatum *>& kernel_data = invocations[i]->m_kernel_data;
                    kernel_data.resize(nargs + arg_offset, nullptr);

                    std::vector<void *>& kernel_parameters = invocations[i]->m_kernel_parameters;
                    kernel_parameters.resize(nargs + arg_offset);

                    // For unmodified kernels
                    std::vector<DatumSegment>& container_segments = invocations[i]->m_container_segments;
                    std::vector<DatumSegment>& container_allocation = invocations[i]->m_container_allocation;

                    SetDevice(i);
#ifndef NDEBUG
                    // Erroneous argument checking
                    std::set<IDatum *> arg_data_input, arg_data_output;
#endif

                    std::vector< std::vector<SegmentCopy> >& deviceSegmentCopies = invocations[i]->m_deviceSegmentCopies;
                    deviceSegmentCopies.resize(total_gpus);

                    // Creates self connections too. Doesn't matter, unused.
                    for (unsigned int j = 0; j < total_gpus; ++j)
                        deviceSegmentCopies[j] = std::vector<SegmentCopy>();

                    // Prepare kernel parameters (skipping the MAPS_MULTIDEF arguments)
                    int inputCtr = 0, outputCtr = 0, constCtr = 0;
                    for (unsigned int p = arg_offset; p < nargs + arg_offset; ++p)
                    {
                        DatumSegment allocated_segment;

                        switch (task.argument_ordering[p - arg_offset])
                        {
                            // Unknown parameter type
                        default:
                            kernel_parameters[p] = nullptr;
                            break;

                            // Constants are copied to each GPU
                        case AT_CONSTANT:
                            kernel_parameters[p] = task.constants[constCtr++].get();
                            break;

                            // Input containers
                        case AT_INPUT:
                            {
                                const auto& input = task.inputs[inputCtr++];
                                IDatum *datum = input.datum;
                                // TODO(later): Conserve copies (only 
                                //              copy union of all 
                                //              necessary information)
                                //              when there is more than one 
                                //              instance of the input.
/*#ifndef NDEBUG
                                if (arg_data_input.find(datum) != arg_data_input.end())
                                {
                                    printf("ERROR: Input datum cannot appear more than once per task\n");
                                    return (taskHandle_t)0;
                                }
                                arg_data_input.insert(datum);
#endif*/
                                if (m_dataFromGraphs.find(datum) != m_dataFromGraphs.end())
                                {
                                    printf("ERROR: Input container already owned by TaskGraph\n");
                                    return (taskHandle_t)0;
                                }

                                // Get the allocated segment
                                if (!m_analyzer.GetSegment(i, datum, allocated_segment))
                                {
                                    printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                                    return (taskHandle_t)0;
                                }

                                Memory mem;

                                // Allocate memory if not allocated yet
                                if (m_buffers[i].find(datum) == m_buffers[i].end())
                                {
                                    mem.ptr = AllocateBuffer(i, datum, allocated_segment, datum->GetElementSize(), mem.stride_bytes);
#ifndef NDEBUG
                                    if (mem.ptr == nullptr && allocated_segment.m_dimensions[0] > 0)
                                    {
                                        printf("ERROR allocating input buffer on GPU %d\n", i);
                                        return (taskHandle_t)0;
                                    }
#endif
                                    m_buffers[i][datum] = mem;
                                }
                                else
                                {
                                    mem = m_buffers[i][datum];
                                }


                                // Memory dependencies and location update
                                ///////////////////////////////////////////

                                // Determine segment that needs to be copied to the device
                                const ContainerSegmentation& segs = input.segmentation;
#ifndef NDEBUG
                                if (segs[i].size() == 0)
                                {
                                    printf("ERROR: No segments necessary for input container\n");
                                    return (taskHandle_t)0;
                                }
#endif
                                DatumSegment ptr_offset = segs[i][0];

                                // The segments that are not covered by our local copy of the memory
                                std::vector<DatumSegment> dirty_segments;

                                // See if the buffer is already (partially) stored on this device
                                if (m_upToDateLocations[i].find(datum) != m_upToDateLocations[i].end())
                                {
                                    // For each necessary segment, see if ANY of the up-to-date segments cover it
                                    for (const auto& necessary_segment : segs[i])
                                    {
                                        bool bCovered = false;
                                        for (const auto& uptodate_segment : m_upToDateLocations[i][datum])
                                            bCovered |= uptodate_segment.Covers(necessary_segment);

                                        // If not, add to dirty segments
                                        if (!bCovered)
                                            dirty_segments.push_back(necessary_segment);

                                        // Compute offset for pointer computation
                                        ptr_offset.BoundingBox(necessary_segment);
                                    }
                                }
                                else
                                {
                                    // All segments need copying
                                    dirty_segments = segs[i];

                                    // Compute offset for pointer computation
                                    for (const auto& necessary_segment : dirty_segments)
                                        ptr_offset.BoundingBox(necessary_segment);
                                }

                                // Schedule copies for all dirty segments
                                if (m_lastLocation.find(datum) == m_lastLocation.end() ||
                                    m_lastLocation[datum].state == LS_HOST)
                                {
                                    // Both above cases mean that the buffer is on the host memory
                                    if (datum->GetDataDimension(0) > 0)
                                    {
#ifndef NDEBUG
                                        if (datum->HostPtr() == nullptr)
                                        {
                                            printf("ERROR: Cannot copy buffer, it is not registered on the host.\n");
                                            return (taskHandle_t)0;
                                        }
#endif

                                        if (!bSkip)
                                        {
                                            // Copy from host to device
                                            for (const DatumSegment& segment : dirty_segments)
                                            {
                                                if (!CopyFromHost(i, datum, allocated_segment, segment, m_streams[i]))
                                                    return (taskHandle_t)0;
                                                m_upToDateLocations[i][datum].push_back(segment);
                                            }
                                        }
                                    }
                                }
                                else
                                {
                                    // Otherwise, the memory is stored on (an)other device(s)
                                    // Act upon last location state
                                    switch (m_lastLocation[datum].state)
                                    {
                                    default:
                                        printf("ERROR: Invalid buffer location for datum(3)\n");
                                        return (taskHandle_t)0;

                                    case LS_NEEDS_AGGREGATION:
                                        // TODO(later)
                                        printf("WARNING: Aggregation to GPU not implemented at this time. "
                                               "Aggregate (gather) to host first and then invoke call.\n");
                                        return (taskHandle_t)0;

                                    case LS_DEVICE:
                                        {
#ifndef NDEBUG
                                            if (m_lastLocation[datum].entries.size() == 0)
                                            {
                                                printf("ERROR: No entries found for buffer although it is reported to be on a device\n");
                                                return (taskHandle_t)0;
                                            }
#endif
                                            // Buffer is entirely on another device
                                            for (const DatumSegment& segment : dirty_segments)
                                            {
                                                // Copy segments from device here
                                                deviceSegmentCopies[m_lastLocation[datum].entries[0].first].
                                                    push_back(std::make_tuple(datum, segment, segment));

                                                // Update up-to-date location
                                                m_upToDateLocations[i][datum].push_back(segment);
                                            }
                                        }
                                        break;

                                    case LS_SEGMENTED:
                                        {
                                            DatumSegment intersection_src, intersection_dst;

                                            // Assuming non-overlapping segments
                                            for (const DatumSegment& segment : dirty_segments)
                                            {
                                                // For each location entry
                                                for (const auto& entry : m_lastLocation[datum].entries)
                                                {
                                                    // Skip entries from same GPU
                                                    if (entry.first == i)
                                                        continue;
                                                    // Skip entries from offline GPUs

                                                    // If the entry intersects with our required segment, compute intersection
                                                    // and copy each of the overlapping segments
                                                    if (IntersectsWith(segment, entry.second, segment.m_borders, datum))
                                                    {
                                                        Intersection(entry.second, segment, datum, intersection_src,
                                                                     intersection_dst);

                                                        deviceSegmentCopies[entry.first].
                                                            push_back(std::make_tuple(datum, intersection_src, intersection_dst));
                                                    }
                                                }

                                                // Update up-to-date location
                                                m_upToDateLocations[i][datum].push_back(segment);
                                            }

                                        }
                                        break;
                                    }
                                }
                                ///////////////////////////////////////////

                                // Compute pointer to argument
                                void *necessary_ptr = allocated_segment.OffsetPtr(ptr_offset.m_offset,
                                                                                  mem.ptr, datum->GetElementSize(),
                                                                                  mem.stride_bytes);
                                mem.ptr = necessary_ptr;

                                if (!bUnmodified)
                                {
                                    // Create MAPS single-GPU container
                                    std::shared_ptr<::maps::IInputContainer> input_container =
                                        input.container_factory->CreateContainer(datum, mem, ptr_offset,
                                                                                 task.segmentation[i]);
                                    m_inputContainers.insert(input_container);

                                    // Finally, set the parameter
                                    kernel_parameters[p] = input_container.get();
                                    kernel_data[p] = datum;

                                    // Unused in Invoke API, used in TaskGraph API
                                    container_segments.push_back(ptr_offset);
                                    container_allocation.push_back(allocated_segment);
                                    container_allocation.back().m_stride_bytes = mem.stride_bytes;
                                }
                                else
                                {
                                    // Just set the parameter to the pointer, and append segment
                                    kernel_parameters[p] = necessary_ptr;
                                    kernel_data[p] = datum;
                                    ptr_offset.m_stride_bytes = mem.stride_bytes;

                                    container_segments.push_back(ptr_offset);
                                    container_allocation.push_back(allocated_segment);
                                    container_allocation.back().m_stride_bytes = mem.stride_bytes;
                                }
                            }
                            break;

                            // Output containers
                        case AT_OUTPUT:
                            {
                                // Compute output container pointers for kernel
                                const auto& output = task.outputs[outputCtr++];
                                IDatum *datum = output.datum;
#ifndef NDEBUG
                                if (arg_data_output.find(datum) != arg_data_output.end())
                                {
                                    printf("ERROR: Output datum cannot appear more than once per task\n");
                                    return (taskHandle_t)0;
                                }
                                arg_data_output.insert(datum);
#endif
                                if (m_dataFromGraphs.find(datum) != m_dataFromGraphs.end())
                                {
                                    printf("ERROR: Output container already owned by TaskGraph\n");
                                    return (taskHandle_t)0;
                                }

                                // Get allocated segment
                                if (!m_analyzer.GetSegment(i, datum, allocated_segment))
                                {
                                    printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                                    return (taskHandle_t)0;
                                }

                                Memory mem;

                                // Allocate memory if not allocated yet
                                if (m_buffers[i].find(datum) == m_buffers[i].end())
                                {
                                    mem.ptr = AllocateBuffer(i, datum, allocated_segment, datum->GetElementSize(), mem.stride_bytes);
#ifndef NDEBUG
                                    if (mem.ptr == nullptr)
                                    {
                                        printf("ERROR allocating output buffer on GPU %d\n", i);
                                        return (taskHandle_t)0;
                                    }
#endif
                                    m_buffers[i][datum] = mem;
                                }
                                else
                                {
                                    mem = m_buffers[i][datum];
                                }

                                // Determine output segments
                                const ContainerSegmentation& segs = output.segmentation;
#ifndef NDEBUG
                                if (segs[i].size() > 1)
                                {
                                    printf("ERROR: Output container should only generate one segment per GPU.\n");
                                    return (taskHandle_t)0;
                                }

                                // If data is outside the scope of analyzed memory, fail
                                if (!allocated_segment.Covers(segs[i][0]))
                                {
                                    printf("ERROR: Output container segment out of bounds. AnalyzeCall was not invoked properly.\n");
                                    return (taskHandle_t)0;
                                }
#endif

                                // Mark location entry to be updated
                                if (locationUpdates.find(datum) == locationUpdates.end())
                                    locationUpdates[datum] = DatumLocation();

                                if (output.type == OCT_REDUCTIVE_CONSTANT ||
                                    output.type == OCT_REDUCTIVE_DYNAMIC)
                                {
                                    locationUpdates[datum].state = (segs.size() == 1 || bInvokeAll) ? LS_DEVICE : LS_NEEDS_AGGREGATION;
                                    m_aggregators[datum] = output.aggregator;
                                }
                                else
                                {
                                    // Assume non-intersecting segments
                                    locationUpdates[datum].state = segs[i][0].Covers(datum) ? LS_DEVICE : LS_SEGMENTED;
                                    m_aggregators.erase(datum);
                                }
                                locationUpdates[datum].entries.push_back(std::make_pair(i, segs[i][0]));

                                ///////////////////////////////////////

                                // Compute pointer to argument
                                void *necessary_ptr = allocated_segment.OffsetPtr(segs[i][0].m_offset,
                                                                                  mem.ptr, datum->GetElementSize(),
                                                                                  mem.stride_bytes);
                                if (!bUnmodified)
                                {
                                    mem.ptr = necessary_ptr;

                                    // Create MAPS single-GPU container
                                    std::shared_ptr<::maps::IOutputContainer> output_container =
                                        output.container_factory->CreateOutputContainer(datum, mem, segs[i][0],
                                                                                        task.segmentation[i]);
                                    m_outputContainers.insert(output_container);

                                    // Finally, set the parameter
                                    kernel_parameters[p] = output_container.get();
                                    kernel_data[p] = datum;

                                    // Unused in Invoke API, used in TaskGraph API
                                    container_segments.push_back(segs[i][0]);
                                    container_segments.back().m_stride_bytes = mem.stride_bytes;
                                    container_allocation.push_back(allocated_segment);
                                    container_allocation.back().m_stride_bytes = mem.stride_bytes;
                                }
                                else
                                {
                                    // Just set the parameter pointer offset and its dimensions
                                    kernel_parameters[p] = necessary_ptr;
                                    kernel_data[p] = datum;

                                    container_segments.push_back(segs[i][0]);
                                    container_segments.back().m_stride_bytes = mem.stride_bytes;
                                    container_allocation.push_back(allocated_segment);
                                    container_allocation.back().m_stride_bytes = mem.stride_bytes;
                                }
                            }
                            break;
                        }
                    }

                    // The reducers support patterns like Adjacency, where logical data is actually composed of 
                    // several real Datum objects (aka virtual datum objects), but we still
                    // want to pass a single argument to the kernel
                    // iterating from the end so that kernel_parameters indexing (erase and insert) work
                    for (auto it = task.reducers.rbegin(); it != task.reducers.rend(); ++it)
                    {
                      // Assuming reducers were injected by order of index and do not overlap

                      int index = std::get<0>(*it);
                      int count = std::get<1>(*it);
                      auto reducer = std::get<2>(*it);

                      std::vector<void*> containers(count);
                      for (size_t j = 0; j < count; ++j)
                        containers[j] = kernel_parameters[arg_offset + index + j];
                                               
                      auto composed = reducer->ComposeContainers(containers, task.segmentation[i]);
                      m_composedContainers.insert(composed);

                      kernel_parameters.erase(kernel_parameters.begin() + arg_offset + index, 
                                              kernel_parameters.begin() + arg_offset + index + count);
                      kernel_parameters.insert(kernel_parameters.begin() + arg_offset + index, composed.get());
                    }

                } // end of GPU iteration

                ++m_taskCounter;

                // If segmented, compute bounding box of each GPU contents to get the updated location
                for (auto& update : locationUpdates)
                {
                    if (m_lastLocation[update.first].state != LS_SEGMENTED || 
                        m_lastLocation[update.first].entries.size() != update.second.entries.size())
                        continue;

                    for (auto& update_seg : update.second.entries)
                    {
                        for (auto& loc : m_lastLocation[update.first].entries)
                        {
                            if (loc.first == update_seg.first)
                                update_seg.second.BoundingBox(loc.second);
                        }
                    }
                }

                // Update last location of output segments (overrides last location entry)
                for (const auto& update : locationUpdates)
                {
                    m_lastLocation[update.first] = update.second;

                    // Override last up-to-date locations related to this output container            
                    for (unsigned int i = 0; i < total_gpus; ++i)
                    {
                        if (m_upToDateLocations[i].find(update.first) != m_upToDateLocations[i].end())
                            m_upToDateLocations[i][update.first].clear();
                        else
                            m_upToDateLocations[i][update.first] = std::vector<DatumSegment>();
                    }
                    for (const auto& entry : update.second.entries)
                        m_upToDateLocations[entry.first][update.first].push_back(entry.second);
                }

                if (!bSkip)
                {
                    EnqueueInvocations(invocations);

                    // Append to task list (for Wait() purposes)
                    m_tasks[m_taskCounter] = task_ptr;
                }



                return bSkip ? 0 : m_taskCounter;
            }


            template<bool async>
            bool GatherInternal(IDatum *datum, bool bReadOnly)
            {
                // Sync with all workers
                for (auto& worker : m_workers) {
                    worker->Sync();
                }

                if (m_lastLocation.find(datum) == m_lastLocation.end())
                {
                    printf("WARNING: Datum not used by tasks or still on host\n");
                    return false;
                }

                DatumLocation& loc = m_lastLocation[datum];

                if (datum->HostPtr() == nullptr)
                {
                    printf("ERROR: Host memory not registered for datum\n");
                    return false;
                }

#ifndef NDEBUG
                if (loc.state != LS_HOST && loc.entries.size() == 0)
                {
                    printf("ERROR: Invalid (empty) location entries for datum\n");
                    return false;
                }
#endif

                // Act according to the last location state of the data
                switch (loc.state)
                {
                default:
                    return false;

                case LS_HOST:
                    // If on host, do nothing
                    return true;

                case LS_DEVICE:
                    {
                        DatumSegment allocated_segment;
                        const auto& entry = loc.entries[0];

                        if (!m_analyzer.GetSegment(entry.first, datum, allocated_segment))
                        {
                            printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                            return false;
                        }

                        // Copy from the first device
                        CopyToHost(entry.first, datum, allocated_segment,
                                   entry.second, m_streams[entry.first], async);
                    }
                    break;

                case LS_SEGMENTED:
                    {
                        DatumSegment allocated_segment;

                        std::vector<cudaEvent_t> events(loc.entries.size());
                        int i = 0;

                        // Copy back to host from each segment, record events in the end
                        for (const auto& entry : loc.entries)
                        {
                            if (!m_analyzer.GetSegment(entry.first, datum, allocated_segment))
                            {
                                printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                                return false;
                            }

                            MAPS_CUDA_CHECK(cudaSetDevice(m_activeGPUs[entry.first]));

                            if (!async)
                                MAPS_CUDA_CHECK(cudaEventCreate(&events[i]));

                            CopyToHost(entry.first, datum, allocated_segment,
                                       entry.second, m_streams[entry.first], true);

                            if (!async)
                                MAPS_CUDA_CHECK(cudaEventRecord(events[i], m_streams[entry.first]));
                            ++i;
                        }

                        i = 0;
                        if (!async)
                        {
                            // Wait for all events to arrive
                            for (cudaEvent_t& ev : events)
                            {
                                MAPS_CUDA_CHECK(cudaSetDevice(m_activeGPUs[loc.entries[i].first]));
                                MAPS_CUDA_CHECK(cudaEventSynchronize(ev));
                                MAPS_CUDA_CHECK(cudaEventDestroy(ev));
                                ++i;
                            }
                        }
                    }
                    break;

                case LS_NEEDS_AGGREGATION:
                    {
                        if (async)
                        {
                            printf("WARNING: Asynchronous aggregation not supported, falling back "
                                   "to synchronous\n");
                        }
#ifndef NDEBUG
                        if (m_aggregators.find(datum) == m_aggregators.end())
                        {
                            printf("ERROR: Aggregator not registered with datum\n");
                            return false;
                        }
                        if (m_aggregators[datum] == nullptr)
                        {
                            printf("ERROR: Null aggregator registered with datum\n");
                            return false;
                        }
#endif
                        // Allocate a temporary host buffer to aggregate with
                        size_t buffersize = datum->GetHostStrideBytes();
                        for (unsigned int dim = 1; dim < datum->GetDataDimensions(); ++dim)
                            buffersize *= datum->GetDataDimension(dim);

                        std::vector<unsigned char> temp_buffer(buffersize);
                        DatumSegment allocated_segment;

                        // Copy back and aggregate
                        for (const auto& entry : loc.entries)
                        {
                            if (!m_analyzer.GetSegment(entry.first, datum, allocated_segment))
                            {
                                printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                                return false;
                            }

                            CopyToHost(entry.first, datum, allocated_segment,
                                       entry.second, m_streams[entry.first], false);

                            m_aggregators[datum]->AggregateToHost(datum, datum->HostPtr(), datum->GetHostStrideBytes(),
                                                                  &temp_buffer[0]);
                        }

                        // Copy aggregated buffer to registered buffer
                        memcpy(datum->HostPtrMutable(), &temp_buffer[0], buffersize);
                    }
                    break;
                }

                if (!bReadOnly)
                {
                    // Update last location of gathered datum (overrides last location entry)
                    loc.state = LS_HOST;
                    loc.entries.clear();

                    // Override last up-to-date locations related to this output container                    
                    for (unsigned int i = 0; i < m_activeGPUs.size(); ++i)
                    {
                        if (m_upToDateLocations[i].find(datum) != m_upToDateLocations[i].end())
                            m_upToDateLocations[i][datum].clear();
                        else
                            m_upToDateLocations[i][datum] = std::vector<DatumSegment>();
                    }
                }

                return true;
            }

        public:

            template<typename D>
            bool Invalidate(D& actual_datum)
            {
                // Signal that the buffer has changed on the host
                // and will therefore have to be copied again

                IDatum *datum = dynamic_cast<IDatum *>(&actual_datum);
                if (datum == nullptr)
                {
                    printf("ERROR: Requested to invalidate a non-datum data structure\n");
                    return false;
                }
                if (datum->HostPtr() == nullptr)
                {
                    printf("ERROR: Host memory not registered for datum\n");
                    return false;
                }

                if (m_lastLocation.find(datum) != m_lastLocation.end())
                {
                    DatumLocation& loc = m_lastLocation[datum];

                    // Override last location entry
                    loc.state = LS_HOST;
                    loc.entries.clear();
                }

                // Override last up-to-date locations related to this output container                    
                for (unsigned int i = 0; i < m_activeGPUs.size(); ++i)
                {
                    if (m_upToDateLocations[i].find(datum) != m_upToDateLocations[i].end())
                        m_upToDateLocations[i][datum].clear();
                    else
                        m_upToDateLocations[i][datum] = std::vector<DatumSegment>();
                }

                return true;
            }

            template<typename First, typename... Args>
            bool Invalidate(First& first, Args&... args)
            {
                return Invalidate(first) && Invalidate(args...);
            }

            template<bool async, typename D>
            bool Gather(D& actual_datum)
            {
                IDatum *datum = dynamic_cast<IDatum *>(&actual_datum);
                if (datum == nullptr)
                {
                    printf("ERROR: Requested to gather a non-datum data structure\n");
                    return false;
                }

                return GatherInternal<async>(datum, false);
            }

            template<bool async, typename First, typename... Args>
            bool Gather(First& first, Args&... args)
            {
                return Gather<async>(first) && Gather<async>(args...);
            }

            template<bool async, typename D>
            bool GatherReadOnly(D& actual_datum)
            {
                IDatum *datum = dynamic_cast<IDatum *>(&actual_datum);
                if (datum == nullptr)
                {
                    printf("ERROR: Requested to gather a non-datum data structure\n");
                    return false;
                }

                return GatherInternal<async>(datum, true);
            }

            template<bool async, typename First, typename... Args>
            bool GatherReadOnly(First& first, Args&... args)
            {
                return GatherReadOnly<async>(first) && GatherReadOnly<async>(args...);
            }

            void Wait(taskHandle_t taskHandle)
            {
                // If already complete, return                
                if (m_tasks.find(taskHandle) == m_tasks.end())
                    return;

                // TODO(later): Wait for a specific task
            }

            void WaitAll()
            {
                // Sync with all workers
                for (auto& worker : m_workers) {
                    worker->Sync();
                }

                // Wait for each task
                for (const auto& iter : m_tasks)
                    Wait(iter.first);
            }
        };

        template<typename... Args>
        static inline void AnalyzeCall(Scheduler& sched, dim3 grid_dims, dim3 block_dims, const Args&... args)
        {
            sched.AnalyzeCall(grid_dims, block_dims, args...);
        }

        template<typename... Args>
        static inline void AnalyzeCallAll(Scheduler& sched, dim3 grid_dims, dim3 block_dims, const Args&... args)
        {
            sched.AnalyzeCallAll(grid_dims, block_dims, args...);
        }

        template<typename Kernel, typename... Args>
        static inline taskHandle_t InvokeDynamicSMem(Scheduler& sched, Kernel kernel, dim3 grid_dims,
                                                     dim3 block_dims, size_t dynamic_smem, const Args&... args)
        {
            return sched.InvokeDynamicSMem(kernel, grid_dims, block_dims, dynamic_smem, args...);
        }

        // Method that converts a function call to a Task
        template<typename Kernel, typename... Args>
        static inline taskHandle_t Invoke(Scheduler& sched, Kernel kernel, dim3 grid_dims,
                                          dim3 block_dims, const Args&... args)
        {
            return sched.InvokeDynamicSMem(kernel, grid_dims, block_dims, 0, args...);
        }

        template<typename D>
        static inline bool Fill(Scheduler& sched, D& actual_datum, int value = 0)
        {
            return sched.Fill(actual_datum, value);
        }

        template<typename... Args>
        static inline bool Gather(Scheduler& sched, Args&... args)
        {
            return sched.Gather<false>(args...);
        }

        template<typename... Args>
        static inline bool GatherReadOnly(Scheduler& sched, Args&... args)
        {
            return sched.GatherReadOnly<false>(args...);
        }

    } // namespace multi

} // namespace maps

#endif // __MAPS_MULTI_SCHEDULER_H
