#!/bin/bash
# This script must be run with sudo.

# Adapted from Caffe: https://github.com/BVLC/caffe/blob/master/scripts/travis/travis_install.sh

set -e

# This ppa is for gflags and glog
add-apt-repository -y ppa:tuleu/precise-backports
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get -y update
apt-get install \
    wget git curl \
    libgflags-dev libgoogle-glog-dev \
    bc gcc-4.9 g++-4.9

# Set GCC 4.9 as the default compiler
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.6 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
update-alternatives --set cc /usr/bin/gcc
update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
update-alternatives --set c++ /usr/bin/g++
update-alternatives --set gcc /usr/bin/gcc-4.9
update-alternatives --set g++ /usr/bin/g++-4.9

# CMake
wget --no-check-certificate http://www.cmake.org/files/v3.2/cmake-3.2.3-Linux-x86_64.sh -O cmake3.sh
chmod +x cmake3.sh
./cmake3.sh --prefix=/usr/ --skip-license --exclude-subdir
rm -f ./cmake3.sh

# Install CUDA 7.5, if needed
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
CUDA_FILE=/tmp/cuda_install.deb
curl $CUDA_URL -o $CUDA_FILE
dpkg -i $CUDA_FILE
rm -f $CUDA_FILE
apt-get -y update
# Install the minimal CUDA subpackages required to test the build.
# For a full CUDA installation, add 'cuda' to the list of packages.
apt-get -y install cuda-core-7-5 cuda-cublas-7-5 cuda-cublas-dev-7-5 cuda-cudart-7-5 cuda-cudart-dev-7-5

# Create CUDA symlink at /usr/local/cuda
# (This would normally be created by the CUDA installer, but we create it
# manually since we did a partial installation.)
ln -s /usr/local/cuda-7.5 /usr/local/cuda
