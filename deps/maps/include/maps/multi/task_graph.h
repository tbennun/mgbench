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

#ifndef __MAPS_MULTI_TASK_GRAPH_H
#define __MAPS_MULTI_TASK_GRAPH_H

// EXPERIMENTAL: This API is experimental and not in use

#include <queue>
#include <vector>

#include "common.h"
#include "scheduler.h"

namespace maps
{
    namespace multi
    {
        enum ActionType
        {
            AT_UNINITIALIZED = 0,

            AT_SET_DEVICE,
            AT_COPY_H2D,       // 2D
            AT_COPY_D2D,       // 1D
            AT_COPY_D2H_SYNC,  // 2D
            AT_COPY_D2H_ASYNC, // 2D
            AT_MEMSET_2D,      // 2D
            AT_MEMSET_1D,      // 1D
            AT_RUN_KERNEL,
            AT_RUN_UNMODIFIED,
            AT_RECORD_EVENT,
            AT_STREAM_WAIT_EVENT,
            AT_AGGREGATE,
        };

        struct Action
        {
            ActionType type;
            unsigned int deviceID; // The device ID in m_activeGPUs

            // Memory operations            
            void *src_ptr, *dst_ptr;
            size_t src_stride_bytes, dst_stride_bytes;
            size_t width_bytes, otherdims;
            unsigned int dstDeviceID;
            size_t total_bytes;
            IDatum *src_datum, *dst_datum; // If not null, we should re-offset
            DatumSegment src_segment, dst_segment; // Re-offset according to
                                                   // this as well


            // Memset parameters
            int memsetValue;

            // Kernels
            const void *kernel_func;
            std::vector<void *> kernel_parameters;
            std::vector<IDatum *> kernel_data;
            dim3 grid_dims, block_dims;
            size_t dsmem;

            // Unmodified routines
            routine_t routine;
            std::vector<uint8_t> context_buffer;
            void *context;
            GridSegment segmentation;
            std::vector<DatumSegment> container_segments, container_allocation;

            cudaStream_t stream;
            cudaEvent_t event;

            Action() : type(AT_UNINITIALIZED), deviceID(0), src_ptr(nullptr), dst_ptr(nullptr),
                src_stride_bytes(0), dst_stride_bytes(0), width_bytes(0), otherdims(0),
                dstDeviceID(0), total_bytes(0), src_datum(nullptr), dst_datum(nullptr),
                memsetValue(0), kernel_func(nullptr), grid_dims(), block_dims(),
                dsmem(0), routine(nullptr), context(nullptr), segmentation(), stream((cudaStream_t)0),
                event((cudaEvent_t)0) { }

            ~Action()
            {
            }
        };

        class TaskGraph : public Scheduler
        {
        protected:
            std::vector<Action> m_queue;
            int m_currentDevice;

            bool m_bFinalized;

            inline bool ExecuteAction(Action& action)
            {
                switch (action.type)
                {
                default:
                    return false;

                case AT_SET_DEVICE:
                    return (cudaSetDevice(this->m_activeGPUs[action.deviceID]) == cudaSuccess);

                case AT_COPY_H2D: // 2D
                    return (cudaMemcpy2DAsync(action.dst_ptr, action.dst_stride_bytes, action.src_ptr, action.src_stride_bytes,
                        action.width_bytes, action.otherdims, cudaMemcpyHostToDevice, action.stream) == cudaSuccess);

                case AT_COPY_D2D: // 1D
                    if (action.event)
                        if (cudaStreamWaitEvent(action.stream, action.event, 0) != cudaSuccess)
                            return false;

                    return (cudaMemcpyPeerAsync(action.dst_ptr, this->m_activeGPUs[action.dstDeviceID], action.src_ptr,
                        this->m_activeGPUs[action.deviceID], action.total_bytes, action.stream) == cudaSuccess);

                case AT_COPY_D2H_ASYNC: // 2D
                    if (action.event)
                        if (cudaStreamWaitEvent(action.stream, action.event, 0) != cudaSuccess)
                            return false;

                    return (cudaMemcpy2DAsync(action.dst_ptr, action.dst_stride_bytes, action.src_ptr,
                        action.src_stride_bytes, action.width_bytes, action.otherdims,
                        cudaMemcpyDeviceToHost, action.stream) == cudaSuccess);

                case AT_COPY_D2H_SYNC: // 2D
                    if (action.event)
                        if (cudaStreamWaitEvent(action.stream, action.event, 0) != cudaSuccess)
                            return false;

                    return (cudaMemcpy2D(action.dst_ptr, action.dst_stride_bytes, action.src_ptr,
                        action.src_stride_bytes, action.width_bytes, action.otherdims,
                        cudaMemcpyDeviceToHost) == cudaSuccess);

                case AT_MEMSET_2D:   // 2D
                    return (cudaMemset2DAsync(action.dst_ptr, action.dst_stride_bytes, action.memsetValue,
                        action.width_bytes, action.otherdims, action.stream) == cudaSuccess);

                case AT_MEMSET_1D:   // 1D
                    return (cudaMemsetAsync(action.dst_ptr, action.memsetValue,
                        action.total_bytes, action.stream) == cudaSuccess);

                case AT_RUN_KERNEL:
                {
                    uint3 block_offset = make_uint3(action.segmentation.offset, 0, 0);
                    dim3 total_grid_dims = action.segmentation.total_grid_dims;

                    // Push the MAPS_MULTIDEF arguments
                    // (unsigned int deviceIdx, dim3 multigridDim, uint3 blockIdx)
                    action.kernel_parameters[0] = &action.deviceID;
                    action.kernel_parameters[1] = &total_grid_dims;
                    action.kernel_parameters[2] = &block_offset;

                    return (cudaLaunchKernel(action.kernel_func, action.segmentation.blocks, action.segmentation.block_dims,
                        &action.kernel_parameters[0], action.dsmem, action.stream) == cudaSuccess);
                }

                case AT_RUN_UNMODIFIED:
                    return action.routine(action.context_buffer.size() > 0 ? &action.context_buffer[0] : action.context,
                                          action.deviceID, action.stream, action.segmentation,
                                          action.kernel_parameters, action.container_segments, action.container_allocation);

                case AT_RECORD_EVENT:
                    return (cudaEventRecord(action.event, action.stream) == cudaSuccess);

                case AT_STREAM_WAIT_EVENT:
                    return (cudaStreamWaitEvent(action.stream, action.event, 0) == cudaSuccess);
                }
            }

        public:

            TaskGraph() : m_currentDevice(-1), m_bFinalized(false) {}
            TaskGraph(const std::vector<unsigned int>& devices) : Scheduler(devices), m_currentDevice(-1), m_bFinalized(false) {}
            TaskGraph(const std::initializer_list<unsigned int>& devices) : Scheduler(devices), m_currentDevice(-1), m_bFinalized(false) {}

            virtual void *AllocateBuffer(unsigned int gpuID, IDatum *datum, const DatumSegment& segment, size_t elementSize,
                                         size_t& strideBytes) override
            {
                // Do nothing in task graphs except determine the correct stride
                if (this->m_allocator->PitchedAllocationEnabled())
                    strideBytes = ::maps::RoundUp((unsigned int)(elementSize * segment.GetDimension(0)), 512U) * 512;
                else
                    strideBytes = elementSize * segment.GetDimension(0);

                return UNALLOCATED_BUFFER;
            }

            virtual bool CopyFromHost(int dstDevice, IDatum *datum, const DatumSegment& allocated_seg,
                                      const DatumSegment& seg, cudaStream_t stream) override
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
#ifndef NDEBUG
                if (!datum)
                    return false;
#endif
                void *src_ptr = datum->HostPtrMutable();
#ifndef NDEBUG
                if (dstDevice >= this->m_activeGPUs.size())
                {
                    printf("ERROR: Invalid device ID\n");
                    return false;
                }
                if (seg.GetDimensions() != datum->GetDataDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
                if (this->m_buffers[dstDevice].find(datum) == this->m_buffers[dstDevice].end())
                {
                    printf("ERROR: H2D - Memory not allocated on device\n");
                    return false;
                }
                if (src_ptr == nullptr)
                {
                    printf("ERROR: H2D - Memory not registered on host (2)\n");
                    return false;
                }
#endif                
                Memory mem_dst = this->m_buffers[dstDevice][datum];
                void *dst_ptr = mem_dst.ptr;

                // Compute real offset
                src_ptr = seg.OffsetPtrBounds(src_ptr, datum, datum->GetElementSize(), datum->GetHostStrideBytes());
                dst_ptr = allocated_seg.OffsetPtr(seg.m_offset, dst_ptr, datum->GetElementSize(), mem_dst.stride_bytes);

                size_t otherdims = 1;
                for (unsigned int i = 1; i < seg.GetDimensions(); ++i)
                    otherdims *= seg.GetDimension(i);

                SetDevice(dstDevice);

                Action action;

                // If segment requires us to set memory, use memset
                if (seg.m_bFill)
                {
                    action.type = AT_MEMSET_2D;
                    action.dst_ptr = dst_ptr;
                    action.dst_datum = datum;
                    action.dst_segment = seg;
                    action.dst_stride_bytes = mem_dst.stride_bytes;
                    action.dstDeviceID = dstDevice;
                    action.memsetValue = seg.m_fillValue;
                    action.width_bytes = datum->GetElementSize() * seg.GetDimension(0);
                    action.otherdims = otherdims;
                    action.stream = stream;
                }
                else // Otherwise, use memcpy
                {
                    action.type = AT_COPY_H2D;
                    action.dst_ptr = dst_ptr;
                    action.dst_datum = datum;
                    action.dst_segment = seg;
                    action.dst_stride_bytes = mem_dst.stride_bytes;
                    action.dstDeviceID = dstDevice;
                    action.src_ptr = src_ptr;
                    action.src_stride_bytes = datum->GetHostStrideBytes();
                    action.width_bytes = datum->GetElementSize() * seg.GetDimension(0);
                    action.otherdims = otherdims;
                    action.stream = stream;
                }

                m_queue.push_back(action);
                return true;
            }


            virtual bool EnqueueCopyToHost(int srcDevice, IDatum *datum, const DatumSegment& allocated_seg,
                                           const DatumSegment& seg, cudaStream_t stream, bool async = true)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
#ifndef NDEBUG
                if (!datum)
                    return false;
#endif
                void *dst_ptr = datum->HostPtrMutable();
#ifndef NDEBUG
                if (srcDevice >= this->m_activeGPUs.size())
                {
                    printf("ERROR: Invalid device ID\n");
                    return false;
                }
                if (seg.GetDimensions() != datum->GetDataDimensions())
                {
                    printf("ERROR: Dimensions must agree\n");
                    return false;
                }
                if (this->m_buffers[srcDevice].find(datum) == this->m_buffers[srcDevice].end())
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
                Memory mem_src = this->m_buffers[srcDevice][datum];
                void *src_ptr = mem_src.ptr;

                // Compute real offset                
                src_ptr = allocated_seg.OffsetPtr(seg.m_offset, src_ptr, datum->GetElementSize(), mem_src.stride_bytes);
                dst_ptr = seg.OffsetPtrBounds(dst_ptr, datum, datum->GetElementSize(), datum->GetHostStrideBytes());

                size_t otherdims = 1;
                for (unsigned int i = 1; i < seg.GetDimensions(); ++i)
                    otherdims *= seg.GetDimension(i);

                SetDevice(srcDevice);

                Action action;

                action.stream = stream;
                action.event = this->m_events[srcDevice];

                if (async)
                {
                    action.type = AT_COPY_D2H_ASYNC;
                    action.dst_ptr = dst_ptr;
                    action.dst_stride_bytes = datum->GetHostStrideBytes();
                    action.src_ptr = src_ptr;
                    action.src_datum = datum;
                    action.src_segment = seg;
                    action.src_stride_bytes = mem_src.stride_bytes;
                    action.deviceID = srcDevice;
                    action.width_bytes = datum->GetElementSize() * seg.GetDimension(0);
                    action.otherdims = otherdims;
                }
                else
                {
                    action.type = AT_COPY_D2H_SYNC;
                    action.dst_ptr = dst_ptr;
                    action.dst_stride_bytes = datum->GetHostStrideBytes();
                    action.src_ptr = src_ptr;
                    action.src_datum = datum;
                    action.src_segment = seg;
                    action.src_stride_bytes = mem_src.stride_bytes;
                    action.deviceID = srcDevice;
                    action.width_bytes = datum->GetElementSize() * seg.GetDimension(0);
                    action.otherdims = otherdims;
                }

                m_queue.push_back(action);
                return true;
            }

            virtual bool CopySegment(unsigned int srcDevice, unsigned int dstDevice, IDatum *datum,
                                     const DatumSegment& segment_src, const DatumSegment& segment_dst,
                                     cudaStream_t stream) override
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
#ifndef NDEBUG
                if (srcDevice == dstDevice)
                {
                    printf("ERROR: Cannot copy segment from device to itself\n");
                    return false;
                }
                if (srcDevice >= this->m_activeGPUs.size() || dstDevice >= this->m_activeGPUs.size())
                {
                    printf("ERROR: Invalid GPU IDs\n");
                    return false;
                }
#endif
                DatumSegment alloc_seg_src, alloc_seg_dst;
                if (!this->m_analyzer.GetSegment(srcDevice, datum, alloc_seg_src))
                {
                    printf("ERROR: No segment allocated at source\n");
                    return false;
                }
                if (!this->m_analyzer.GetSegment(dstDevice, datum, alloc_seg_dst))
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
                if (this->m_buffers[srcDevice].find(datum) == this->m_buffers[srcDevice].end())
                {
                    printf("ERROR: Memory not allocated on device\n");
                    return false;
                }
                if (this->m_buffers[dstDevice].find(datum) == this->m_buffers[dstDevice].end())
                {
                    printf("ERROR: Memory not allocated on device (2)\n");
                    return false;
                }
#endif
                Memory mem_src, mem_dst;
                mem_src = this->m_buffers[srcDevice][datum];
                mem_dst = this->m_buffers[dstDevice][datum];
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

                Action action;

                action.stream = stream;

                if (segment_dst.m_bFill) // If segment requires us to set memory, use memset
                {
                    SetDevice(dstDevice);

                    action.type = AT_MEMSET_1D;
                    action.dst_ptr = ptr_dst;
                    action.dst_datum = datum;
                    action.dst_segment = segment_dst;
                    action.dstDeviceID = dstDevice;
                    action.memsetValue = segment_dst.m_fillValue;
                    action.total_bytes = bytes;
                }
                else
                {
                    action.event = this->m_events[srcDevice];

                    action.type = AT_COPY_D2D;
                    action.dst_ptr = ptr_dst;
                    action.dst_datum = datum;
                    action.dst_segment = segment_dst;
                    action.dstDeviceID = dstDevice;
                    action.src_ptr = ptr_src;
                    action.src_datum = datum;
                    action.src_segment = segment_src;
                    action.deviceID = srcDevice;
                    action.total_bytes = bytes;
                }

                m_queue.push_back(action);
                return true;
            }

            virtual void SetDevice(unsigned int gpuID) override
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return;
                }

                int actualGPU = (int)this->m_activeGPUs[gpuID];
                if (m_currentDevice == actualGPU)
                    return;

                m_currentDevice = actualGPU;

                Action action;
                action.type = AT_SET_DEVICE;
                action.deviceID = gpuID;

                m_queue.push_back(action);
            }

            virtual bool LaunchKernel(const void *kernel, int deviceIdx, const GridSegment& segmentation,
                                      std::vector<void *>& kernel_parameters, size_t dsmem,
                                      const std::vector<DatumSegment>& container_segments, std::vector<IDatum *>& kernel_data) override
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }

                Action action;

                action.type = AT_RUN_KERNEL;
                action.kernel_func = kernel;
                action.deviceID = deviceIdx;
                action.segmentation = segmentation;
                action.kernel_parameters = kernel_parameters;
                action.container_segments = container_segments;
                action.kernel_data = kernel_data;
                action.dsmem = dsmem;
                action.stream = this->m_streams[deviceIdx];

                m_queue.push_back(action);
                return true;
            }

            virtual bool CallUnmodifiedRoutine(routine_t routine, void *context, std::vector<uint8_t>& copied_context, int deviceIdx, const GridSegment& segmentation,
                                               std::vector<void *>& kernel_parameters,
                                               const std::vector<DatumSegment>& container_segments,
                                               std::vector<IDatum *>& kernel_data,
                                               const std::vector<DatumSegment>& container_allocation) override
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }

                Action action;

                action.type = AT_RUN_UNMODIFIED;
                action.routine = routine;
                action.context = context;
                action.context_buffer = copied_context;
                action.deviceID = deviceIdx;
                action.segmentation = segmentation;
                action.kernel_parameters = kernel_parameters;
                action.container_segments = container_segments;
                action.container_allocation = container_allocation;
                action.kernel_data = kernel_data;
                action.stream = this->m_streams[deviceIdx];

                m_queue.push_back(action);
                return true;
            }

            virtual bool RecordEvent(int deviceIdx, int laneIdx = 0)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }

                Action action;

                action.type = AT_RECORD_EVENT;
                action.deviceID = deviceIdx;

                action.event = this->m_events[deviceIdx];
                action.stream = this->m_streams[deviceIdx];

                m_queue.push_back(action);
                return true;
            }
        protected:
            virtual void EnqueueInvocations(std::vector< std::shared_ptr<Scheduler::Invocation> >& invocations) override
            {
                this->RunInvocationsSequentially(invocations);
            }

        public:
            template<typename Kernel, typename... Args>
            inline bool Enqueue(Kernel kernel, dim3 grid_dims,
                                dim3 block_dims, const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCall(grid_dims, block_dims, args...);
                if (this->InvokeInternal(kernel, grid_dims, block_dims, 0, false, nullptr, nullptr, 0, false, false, args...))
                    return true;
                return false;
            }

            template<typename Kernel, typename... Args>
            inline bool EnqueueDynamicSMem(Kernel kernel, dim3 grid_dims, dim3 block_dims,
                                           size_t dynamic_smem, const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCall(grid_dims, block_dims, args...);
                if (this->InvokeInternal(kernel, grid_dims, block_dims, dynamic_smem, false, nullptr, nullptr, 0, false, false, args...))
                    return true;
                return false;
            }

            template<typename... Args>
            inline bool EnqueueUnmodified(routine_t routine, void *context, dim3 work_dims,
                                          const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCall(work_dims, dim3(), args...);
                if (this->InvokeInternal(nullptr, work_dims, dim3(), 0, true, routine, context, 0, false, false, args...))
                    return true;
                return false;
            }

            template<typename... Args>
            inline bool EnqueueUnmodifiedCopyContext(routine_t routine, void *context, size_t contextSize, dim3 work_dims,
                                                     const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCall(work_dims, dim3(), args...);
                if (this->InvokeInternal(nullptr, work_dims, dim3(), 0, true, routine, context, contextSize, false, false, args...))
                    return true;
                return false;
            }

            template<typename Kernel, typename... Args>
            inline bool EnqueueAll(Kernel kernel, dim3 grid_dims,
                                   dim3 block_dims, const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCallAll(grid_dims, block_dims, args...);
                if (this->InvokeInternal(kernel, grid_dims, block_dims, 0, false, nullptr, nullptr, 0, true, false, args...))
                    return true;
                return false;
            }

            template<typename Kernel, typename... Args>
            inline bool EnqueueAllDynamicSMem(Kernel kernel, dim3 grid_dims, dim3 block_dims,
                                              size_t dynamic_smem, const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCallAll(grid_dims, block_dims, args...);
                if (this->InvokeInternal(kernel, grid_dims, block_dims, dynamic_smem, false, nullptr, nullptr, 0, true, false, args...))
                    return true;
                return false;
            }

            template<typename... Args>
            inline bool EnqueueAllUnmodified(routine_t routine, void *context, dim3 work_dims,
                                             const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCallAll(work_dims, dim3(), args...);
                if (this->InvokeInternal(nullptr, work_dims, dim3(), 0, true, routine, context, 0, true, false, args...))
                    return true;
                return false;
            }

            template<typename... Args>
            inline bool EnqueueAllUnmodifiedCopyContext(routine_t routine, void *context, size_t context_size, dim3 work_dims,
                                                        const Args&... args)
            {
                if (m_bFinalized)
                {
                    printf("ERROR: Cannot enqueue to an already finalized task graph\n");
                    return false;
                }
                this->AnalyzeCallAll(work_dims, dim3(), args...);
                if (this->InvokeInternal(nullptr, work_dims, dim3(), 0, true, routine, context, context_size, true, false, args...))
                    return true;
                return false;
            }

            // A subset of Scheduler::Gather (due to same functionality, as well as to allow
            // external gathers on TaskGraphs)
            template<typename D>
            bool EnqueueGather(D& actual_datum)
            {
                IDatum *datum = dynamic_cast<IDatum *>(&actual_datum);
                if (datum == nullptr)
                {
                    printf("ERROR: Requested to gather a non-datum data structure\n");
                    return false;
                }
                if (this->m_lastLocation.find(datum) == this->m_lastLocation.end())
                {
                    printf("WARNING: Datum not used by tasks or still on host\n");
                    return true;
                }

                DatumLocation loc = this->m_lastLocation[datum];

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

                    if (!this->m_analyzer.GetSegment(entry.first, datum, allocated_segment))
                    {
                        printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                        return false;
                    }

                    // Copy from the first device
                    EnqueueCopyToHost(entry.first, datum, allocated_segment,
                                      entry.second, this->m_streams[entry.first]);
                }
                return true;

                case LS_SEGMENTED:
                {
                    DatumSegment allocated_segment;

                    std::vector<cudaEvent_t> events(loc.entries.size());

                    // Copy back to host from each segment, record events in the end
                    for (int i = 0; i < loc.entries.size(); ++i)
                    {
                        unsigned int srcDev = loc.entries[i].first;
                        if (!this->m_analyzer.GetSegment(srcDev, datum, allocated_segment))
                        {
                            printf("ERROR: Segment not found in analyzer. AnalyzeCall was not invoked properly.\n");
                            return false;
                        }

                        EnqueueCopyToHost(srcDev, datum, allocated_segment,
                                          loc.entries[i].second, this->m_streams[srcDev]);
                    }
                }
                return true;

                case LS_NEEDS_AGGREGATION:
                    // TODO(later): Use stream callbacks
                    printf("ERROR: Queued aggregation not supported yet.\n");
                    return false;
                }
            }

            template<typename First, typename... Args>
            bool EnqueueGather(First& first, Args&... args)
            {
                return EnqueueGather(first) && EnqueueGather(args...);
            }


            bool Finalize()
            {
                m_bFinalized = true;

                DatumSegment allocated_segment;

                // Allocate all necessary buffers
                for (unsigned int i = 0; i < this->m_activeGPUs.size(); ++i)
                {
                    for (auto&& buff : this->m_buffers[i])
                    {
                        IDatum *datum = buff.first;

                        // Get allocated segment
                        if (!this->m_analyzer.GetSegment(i, datum, allocated_segment))
                        {
                            printf("ERROR: Segment not found in analyzer. Enqueue was not invoked properly.\n");
                            return false;
                        }

                        Memory mem;
                        mem.ptr = this->m_allocator->Allocate(this->m_activeGPUs[i], allocated_segment, datum->GetElementSize(), mem.stride_bytes);

#ifndef NDEBUG
                        if (mem.stride_bytes != buff.second.stride_bytes)
                        {
                            printf("ERROR: INCORRECTLY DETERMINED STRIDE: Ours: %d, CUDA: %d\n",
                                   buff.second.stride_bytes, mem.stride_bytes);
                        }
#endif

                        buff.second = mem;
                    }
                }

                for (auto&& action : m_queue)
                {
                    // Re-offset unallocated buffers in memory copies
                    if (action.src_ptr && action.src_datum)
                    {
                        Memory mem = this->m_buffers[action.deviceID][action.src_datum];
                        if (!this->m_analyzer.GetSegment(action.deviceID, action.src_datum, allocated_segment))
                        {
                            printf("ERROR: Segment not found in analyzer. Enqueue was not invoked properly.\n");
                            return false;
                        }

                        action.src_ptr = allocated_segment.OffsetPtr(action.src_segment.m_offset, mem.ptr,
                                                                     action.src_datum->GetElementSize(),
                                                                     mem.stride_bytes);
                    }
                    if (action.dst_ptr && action.dst_datum)
                    {
                        Memory mem = this->m_buffers[action.dstDeviceID][action.dst_datum];
                        if (!this->m_analyzer.GetSegment(action.dstDeviceID, action.dst_datum, allocated_segment))
                        {
                            printf("ERROR: Segment not found in analyzer. Enqueue was not invoked properly.\n");
                            return false;
                        }

                        action.dst_ptr = allocated_segment.OffsetPtr(action.dst_segment.m_offset, mem.ptr,
                                                                     action.dst_datum->GetElementSize(),
                                                                     mem.stride_bytes);
                    }

                    int segmentInd = 0;
                    // Re-offset unallocated buffers in kernel and routine calls
                    for (unsigned int p = 0; p < action.kernel_parameters.size(); ++p)
                    {
                        if (action.kernel_data[p] != nullptr)
                        {
                            DatumSegment& seg = action.container_segments[segmentInd++];
                            Memory mem = this->m_buffers[action.deviceID][action.kernel_data[p]];
                            if (!this->m_analyzer.GetSegment(action.deviceID, action.kernel_data[p], allocated_segment))
                            {
                                printf("ERROR: Segment not found in analyzer. Enqueue was not invoked properly.\n");
                                return false;
                            }

                            // In kernels, the pointer is within the container
                            if (action.type == AT_RUN_KERNEL)
                            {
                                IContainer *container = (IContainer *)action.kernel_parameters[p];
#ifndef NDEBUG
                                if (container == nullptr)
                                {
                                    printf("ERROR in argument %d: Datum given but parameter is not a container\n", p + 1);
                                    return false;
                                }
#endif
                                container->m_ptr = allocated_segment.OffsetPtr(seg.m_offset, mem.ptr,
                                                                               action.kernel_data[p]->GetElementSize(),
                                                                               mem.stride_bytes);
                            }
                            else if (action.type == AT_RUN_UNMODIFIED)  // Unmodified routines
                            {
                                action.kernel_parameters[p] = allocated_segment.OffsetPtr(seg.m_offset, mem.ptr,
                                                                                          action.kernel_data[p]->GetElementSize(),
                                                                                          mem.stride_bytes);
                            }
                        }
                    }
                }

                return true;
            }

            bool CallGraph()
            {
                for (unsigned int i = 0; i < m_queue.size(); ++i)
                {
#ifdef NDEBUG
                    ExecuteAction(m_queue[i]);
#else
                    if (!ExecuteAction(m_queue[i]))
                    {
                        printf("ERROR: Failed to execute action %d: type = %d\n", i, (int)m_queue[i].type);
                        return false;
                    }

                    MAPS_CUDA_CHECK(cudaDeviceSynchronize());
#endif
                }
                return true;
            }
        };

    }  // namespace multi
}  // namespace maps

#endif  // __MAPS_MULTI_TASK_GRAPH_H
