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

cmake_minimum_required (VERSION 2.8)
project (mgbench)

############## DEPENDENCIES ##############

find_package(CUDA 7 REQUIRED)

add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

include_directories(${PROJECT_SOURCE_DIR}/deps/maps/include)

################# FLAGS ##################

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-lineinfo;-Xcompiler;-ggdb;-std=c++11)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG;-std=c++11)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (WIN32 AND NOT CYGWIN)
  set(EXTRA_LIBS "")
else()
  set(EXTRA_LIBS "pthread")
endif()

#########################################

# Level 0 (diagnostics)

cuda_add_executable(numgpus src/L0/numgpus.cpp)
cuda_add_executable(devinfo src/L0/devinfo.cpp)

# Level 1 (microbenchmarks)

cuda_add_executable(halfduplex src/L1/halfduplex.cpp)
target_link_libraries(halfduplex gflags-static ${EXTRA_LIBS})

cuda_add_executable(fullduplex src/L1/fullduplex.cpp)
target_link_libraries(fullduplex gflags-static ${EXTRA_LIBS})

cuda_add_executable(uva src/L1/uva.cu)
target_link_libraries(uva gflags-static ${EXTRA_LIBS})

cuda_add_executable(scatter src/L1/scatter.cpp)
target_link_libraries(scatter gflags-static ${EXTRA_LIBS})

# Level 2 (micro-applications)

cuda_add_executable(gol src/L2/gol/golsample.cu src/L2/gol/main.cpp)
target_link_libraries(gol gflags-static ${EXTRA_LIBS})

cuda_add_executable(sgemm src/L2/sgemm/sgemm.cu src/L2/sgemm/main.cpp)
cuda_add_cublas_to_target(sgemm)
target_link_libraries(sgemm gflags-static ${EXTRA_LIBS})
