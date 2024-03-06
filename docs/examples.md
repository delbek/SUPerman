# Example Commands

This document provides practical examples of how to run SUPerman with various configurations. These examples demonstrate the versatility of SUPerman in handling different matrix types, computational preferences, and hardware settings.

## Basic Usage Examples

1. **Running on GPU with Default Algorithm**
    ```bash
    ./gpu_perman ex.mtx -p 1
    ```
   Calls `gpu_perman_xlocal()` for GPU device 0.

2. **Running on GPU with Binary Matrix Conversion**
    ```bash
    ./gpu_perman ex.mtx -p 1 -b
    ```
   Calls `gpu_perman_xlocal()` for GPU device 0, converting the matrix to binary format.

3. **Sparse Matrix on GPU with Binary Conversion**
    ```bash
    ./gpu_perman ex.mtx -s -p 1 -b
    ```
   Calls `gpu_perman_xlocal_sparse()` for GPU device 0, converting the matrix to binary format.

4. **Sparse Matrix on Specific GPU Device**
    ```bash
    ./gpu_perman ex.mtx -s -p 1 -l2
    ```
   Calls `gpu_perman_xlocal_sparse()` for GPU device 2.

## Advanced Configuration Examples

5. **Sparse Matrix Calculation on CPU with Multiple Threads**
    ```bash
    ./gpu_perman ex.mtx -s -p 1 -c -t 32
    ```
   Calls `parallel_perman_sparse()` using 32 CPU threads.

6. **Multi-GPU Execution**
    ```bash
    ./gpu_perman ex.mtx -p 5 -d 4
    ```
   Executes `gpu_perman64_xshared_lbc_mshared_multigpu()` on 4 GPUs.

7. **Hybrid Multi-GPU and Multi-CPU Execution with Sparse Matrix**
    ```bash
    ./gpu_perman ex.mtx -s -p 5 -d 4 -c -t 64
    ```
   Runs `gpu_perman64_xshared_lbc_mshared_multigpucpu_chunks_sparse()` on 4 GPUs and 64 CPU threads.

## Distributed Computing Examples

8. **Distributed GPUs Execution**
    ```bash
    mpi_run -np 16 ./mpi_perman ex.mtx -s -p 5 -d 4 -c -t 64
    ```
   Invokes `gpu_perman64_xregister_lbc_plainmatrix_mshared_mpi()` to run on 16 distributed GPUs.

## High Precision Calculation Examples

9. **High Precision Sparse Matrix Calculation on CPU**
    ```bash
    ./cpu_perman ex.mtx -s -p 1 -c -t 32 -q -v
    ```
   Executes `parallel_perman_sparse()` with 32 threads, using a 128-bit data type for both calculation and storage.

These examples showcase just a fraction of SUPerman's capabilities, providing a starting point for users to explore the tool's full potential. Adjust the commands according to your specific needs, matrix types, and available hardware resources.
