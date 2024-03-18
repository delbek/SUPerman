# Implementations

SUPerman offers a wide range of algorithms for computing the permanent of matrices, catering to different needs such as exact or approximate calculations, and support for both dense and sparse matrix representations. The following table outlines the available implementations, their key features, applicable flags, and assumptions.

## Available Algorithms

| Implementation                                           | Device     | Calculation | Matrix Type | CPU Flag | Threads | GPU Flag | Algo ID | Approximation | Sparse | Preprocessing | Assumption     |
|----------------------------------------------------------|------------|-------------|-------------|----------|---------|----------|---------|---------------|--------|---------------|----------------|
| `parallel_perman64()`                                    | CPU        | Exact       | Dense       | Yes      | 1-64    | No       | 1       | No            | No     | None          |                |
| `parallel_perman64_sparse()`                             | CPU        | Exact       | Sparse      | Yes      | 1-64    | No       | 1       | No            | Yes    | Optional (-r) | -r1            |
| `parallel_skip_perman64_w()`                             | CPU        | Exact       | Sparse      | Yes      | 1-64    | No       | 2       | No            | Yes    | Optional (-r) | -r2            |
| `parallel_skip_perman64_w_balanced()`                    | CPU        | Exact       | Sparse      | Yes      | 1-64    | No       | 3       | No            | Yes    | Optional (-r) | -r2            |
| `gpu_perman64_xglobal()`                                 | GPU        | Exact       | Dense       | No       | -       | Yes      | 21      | No            | No     | None          |                |
| `gpu_perman64_xlocal()`                                  | GPU        | Exact       | Dense       | No       | -       | Yes      | 1       | No            | No     | None          |                |
| `gpu_perman64_xshared_lbc_mshared_multigpucpu_chunks()`  | GPU+CPU    | Exact       | Dense       | Optional | 1-64    | Yes      | 7       | No            | No     | None          |                |
| `gpu_perman64_xlocal_sparse()`                           | GPU        | Exact       | Sparse      | No       | -       | Yes      | 1       | No            | Yes    | Optional (-r) | -r1            |
| `gpu_perman64_xshared_sparse()`                          | GPU        | Exact       | Sparse      | No       | -       | Yes      | 2       | No            | Yes    | Optional (-r) | -r1            |

### Legend

- **Device**: The computing device(s) the algorithm runs on (CPU, GPU, or distributed GPUs).
- **Calculation**: Specifies whether the algorithm performs exact or approximate calculations.
- **Matrix Type**: Indicates whether the algorithm is designed for dense or sparse matrices.
- **CPU Flag (`-c`)**: Indicates if the algorithm can run on CPU. "Yes" means CPU is supported; "Optional" indicates the algorithm can run on both CPU and GPU; "No" indicates GPU only.
- **Threads (`-t`)**: The range of CPU threads that can be utilized.
- **GPU Flag (`-g`)**: Specifies if the algorithm supports execution on GPU.
- **Algo ID (`-p`)**: The identifier used to select the algorithm via command-line.
- **Sparse (`-s`)**: Denotes whether the algorithm is optimized for sparse matrices.
- **Preprocessing (`-r`)**: Specifies if preprocessing can be applied, and under what conditions.
- **Assumption**: Any specific assumptions or conditions unique to the algorithm.

This table provides a comprehensive overview of the algorithms available in SUPerman for computing matrix permanents. Users can select the most suitable algorithm based on their specific requirements, including the type of matrix, desired accuracy, and available hardware resources.
