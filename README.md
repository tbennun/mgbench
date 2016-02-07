MGBench: Multi-GPU Computing Benchmark Suite
==========================================

[![Build Status](https://travis-ci.org/tbennun/mgbench.svg?branch=master)](https://travis-ci.org/tbennun/mgbench)

This set of applications test the performance, bus speed, power efficiency and correctness of a multi-GPU node.

It is comprised of Level-0 tests (diagnostic utilities), Level-1 tests (microbenchmarks), and Level-2 tests (micro-applications).

Requirements
------------

CMake 2.8 or higher.

CUDA 7.0 or higher. If not installed in the default path, make sure to set the `CUDA_BIN_PATH` environment variable to the CUDA root directory (e.g. `/usr/local/cuda-7.0`).


Instructions
------------

To build, execute the included `build.sh` script. Alternatively, create a 'build' directory and run `cmake ..` within, followed by `make`.

To run the tests, execute the included `run.sh` script. The results will then be placed in the working directory.

Tests
-----

A full list of the tests, their purpose, and command-line arguments can be found [here](https://github.com/tbennun/mgbench/blob/master/TESTS.md).

License
-------

MGBench is published under the New BSD license, see [LICENSE](https://github.com/tbennun/mgbench/blob/master/LICENSE).


Included Software and Licenses
------------------------------

The following dependencies are included within the repository for ease of compilation:

* [gflags](https://github.com/gflags/gflags): [New BSD License](https://github.com/tbennun/mgbench/blob/master/deps/gflags/COPYING.txt). Copyright (c) 2006, Google Inc. All rights reserved.

* [MAPS](https://github.com/maps-gpu/MAPS): [New BSD License](https://github.com/tbennun/mgbench/blob/master/deps/maps/LICENSE). Copyright (c) 2015, A. Barak. All rights reserved.


