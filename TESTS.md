List of Tests
=============

Level-1
-------

* Half-duplex: Tests inter-GPU bandwidth by copying data from each GPU to the rest of the GPUs.
  * Special flags:
    * `--from`: Specify only one GPU to copy from (or -1 for all GPUs)
    * `--to`: Specify a single target GPU to copy to (or -1 for all GPUs)

* Scaling: Tests performance degradation as a result of multi-GPU utilization. Runs SGEMM (see Level-2 tests) in a mode that performs the same operation on all GPUs. The time should be roughly the same for any number of GPUs.


Level-2
-------

* Matrix Multiplication (sgemm): Performs multi-GPU matrix multiplication (using CUBLAS), aggregating the results to the host without performing inter-GPU communications.
  * Modes:
    * Correctness: Runs SGEMM with CPU regression for a small amount of iterations to verify the results.
    * Performance: Runs SGEMM without regression, averaging multiplication time over a large amount of repetitions to obtain accurate performance. Scaling should be near-linear.

* Game of Life (gol): Simple stencil operator that tests multi-GPU correctness as well as inter-GPU communications.
  * Modes:
    * Correctness: Runs GoL with CPU regression for a small amount of iterations to verify the results.
    * Performance: Runs GoL without regression, averaging iteration time over a large amount of iterations to obtain accurate performance. Scaling should be near-linear, albeit less than the scaling of SGEMM.
    * Single-GPU: Runs GoL with regression tests on each of the GPUs separately, pinpointing faulty devices.
  * Special flags:
    * `--save_images`: Saves the two images in case the regression test failed (default: true).



Test Flags
----------

* `--regression`: Performs regression tests against a correct CPU version (default: true).
* `--num_gpus`: Override the maximal number of GPUs to run on, simulating a larger number of GPUs if necessary using Round-Robin on the available GPUs (default: -1, use all GPUs).
* `--startwith`: Override the minimal number of GPUs to run on (default: 0).
* `--gpuoffset`: Offsets the GPU IDs so that a single GPU, or subsequences of the GPUs, can be chosen (default: 0).

