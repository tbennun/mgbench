MAPS: GPU Optimization and Memory Abstraction Framework
=======================================================

[![Build Status](https://travis-ci.org/maps-gpu/MAPS.svg?branch=master)](https://travis-ci.org/maps-gpu/MAPS)

MAPS is a header-only C++ CUDA template library for automatic optimization of 
GPU kernels and transparent partitioning of multi-GPU tasks. 
It uses memory access patterns to provide near-optimal 
performance while maintaining code simplicity.

For more information, see the framework website at:
http://www.cs.huji.ac.il/project/maps/


Requirements
------------

CUDA 7.0 or higher.

gflags (for command-line arguments in samples): https://github.com/gflags/gflags

Google Test (for unit tests): https://github.com/google/googletest


Installation
------------

To compile code with MAPS, use the includes under the "include" directory.

It is generally recommended to include MAPS using the 
all-inclusive header (from .cu files only):

``` cpp
#include <maps/maps.cuh>
```


Samples
-------

Code samples are available under the "samples" directory. To compile,
either use Visual Studio on Windows or CMake on other platforms (http://www.cmake.org/)
