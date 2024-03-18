# Usage Guide for SUPerman

SUPerman is a high-performance computational tool designed for the efficient calculation of matrix permanents utilizing CPUs, GPUs, and distributed computing environments. This guide outlines the various executable arguments available to customize your computation tasks.

## Executable Arguments Overview

SUPerman can be executed with a range of arguments to specify the computation's nature, the hardware to be used, and the type of algorithm, among other settings. Below is a detailed explanation of each argument:

### Usage

`cpu_perman [FILE] [OPTION]...`
`gpu_perman [FILE] [OPTION]...`
`mpi_perman [FILE] [OPTION]...`
where `FILE` is the path to the matrix.

### Algorithm Selection

- `-p, --perman <id>`: Selects the algorithm for permanent calculation. Default is `0`.
- `-s, --sparse`: If set, the sparse algorithm is used for computation. By default, a dense algorithm is chosen.
- `-b, --binary`: Treats the input matrix as binary, where all non-zero values are considered to be 1.

### Hardware Configuration

- `-g, --gpu`: Executes the permanent calculation on the GPU. This is the default mode unless `-c` is specified.
- `-c, --cpu`: Forces the computation to run on the CPU. If used together with `-g`, a hybrid algorithm is chosen.
- `-d, --device <number>`: Specifies the number of devices to be used in a multi-GPU algorithm. Default is `2`.
- `-t, --threads <number>`: Determines the number of threads to use for CPU computations. Default is `16`.
- `-l, --gpu-id <id>`: Chooses a specific GPU with the given ID to run a single GPU algorithm. Default is `0`.

### Precision and Data Representation

- `--calculate-32bit`: Uses a 32-bit data type for calculation.
- `--calculate-128bit`: Uses a 128-bit data type for calculation, available only on the CPU.
- `--storage-32bit`: Uses a 32-bit data type for storing matrix data.
- `--storage-128bit`: Uses a 128-bit data type for storage, available only on the CPU.

### Algorithm Customization

- `-r, --preprocessing <value>`: Applies preprocessing (1: SortOrder, 2: SkipOrder). Default is no preprocessing (`0`).
- `--scale <value>`: Scales the input matrix to the specified value. Default is `1.0`.
- `-y, --scale-intervals <number>`: Defines scale intervals for a scaling approximation algorithm. Default is `4`.
- `-z, --scale-count <number>`: Indicates the number of times to scale for a scaling approximation algorithm. Default is `5`.

### Miscellaneous

- `-k <repetitions>`: Runs the chosen algorithm a specified number of times independently. Default is `1`.
- `--multiply-dim <multiplier>`: Multiplies CUDA runtime chosen grid dimension for GPU algorithms. Default is `1`.
- `--compress`: Enables compression to reduce memory footprint.
