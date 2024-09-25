# GPGPU

## Authors
* Pierre-Louis Delcroix
* Marc Lagoin
* Scott Tallec
* Daniel Rosa

## Description
This project is an implementation of object detection using Choquet's integral in C++ using CUDA.

## Requirements
* CUDA
* ffmpeg

## Setup
* Clone the repository
* Download the dataset and extract it at the root of the directory

From the root of the directory:

```bash
mkdir build
cd build
cmake .
make
```

## Usage
```bash
./cpu
./cpu_optimized
./gpu
./gpu_optimized
```


