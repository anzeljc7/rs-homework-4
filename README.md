# Fourth homework assignment

## Task 1: Analyzing the performance of GPU kernels in GEM5 (4 points)


In this task, you will evaluate the performance characteristics of a histogram computation kernel using the GEM5 GPU simulator. The histogram kernel is an algorithm that processes an input image and computes the frequency distribution of pixel values. The input image `lena.jpg` is provided in the `histogram/` directory. The histogram is computed over 256 bins, corresponding to the pixel values ranging from 0 to 255. 

You are tasked with analyzing and comparing the performance of two kernel implementations:

- Naive implementation: This version updates a single histogram stored in global memory, resulting in frequent memory contention.

- Optimized implementation with privatization: In this version, each group of threads maintains a private copy of the histogram. After the parallel computation completes, these private histograms are merged into a single global histogram.

Both kernel versions are provided in the `histogram/` directory. The implementations are written in HIP and are configured to run within the GEM5 GPU simulation environment.

To build and simulate the kernels, use the provided Makefile in conjunction with the `make_apptainer.sh` script. This script automates the build process inside the pre-configured container environment for GEM5 GPU simulation.

### Experiment Setup:

1. Set the number of compute units to 2, 4, and 8.
2. Run both of the histogram kernels for each configuration.
3. To observe the impact of false sharing, observe following metrics:
   - Mean load latency (`loadLatencyDist::mean`)
   - Avergage number of executed vector ALU instructions (`vALUInsts`)
   - Average number of access to the shared memory (`ldsBankAccess`)
   - Average number of cycles (`totalCycles`)
   - Average number of vector per cycle (`vpc`)

> Note: When analyzing the results in the `stats.txt` file, you will notice that each performance metric appears **twice**. The **first occurrence** of each metric corresponds to the period **during GPU kernel execution**.
The **second occurrence** is collected **after the GPU kernel has completed**. For the task at hand, you should **focus on the first set of statistics**, as they reflect the actual behavior and performance characteristics of the GPU kernel while it is executing.



## Task 2: Implementing Matrix transpose using Shared Memory (3 points)

In this task, you will evaluate the performance characteristics of two matrix transpose kernels using the GEM5 GPU simulator. The matrix transpose operation rearranges an input matrix A of size N × N such that the output matrix B satisfies B[col][row] = A[row][col] for all valid indices. The input matrix is a 512 × 512 array of single-precision floating-point values, initialized with sequential values. Both kernel implementations are provided in the `transpose/` directory and are written in HIP, configured to run within the GEM5 GPU simulation environment.
You are tasked with analyzing and comparing the performance of two kernel implementations:

- Naive implementation: This version performs the transpose by reading each element from the input matrix in row-major order and writing it directly to the transposed position in the output matrix. 
- Shared memory implementation: This version uses a tile of Local Data Share (LDS) memory of size 16 × 17 elements as a staging buffer. 

To build and simulate the kernels, use the provided Makefile in conjunction with the make_apptainer.sh script. This script automates the build process inside the pre-configured container environment for GEM5 GPU simulation.

### Experiment Setup:

1. Set the number of compute units to 2, 4, and 8.
2. Run the both of the kernels for each configuration.
3. To compare the performance of the kernels, observe following metrics:
   - Mean load latency (`loadLatencyDist::mean`)
   - Avergage number of executed vector ALU instructions (`vALUInsts`)
   - Average number of access to the shared memory (`ldsBankAccess`)
   - Average number of cycles (`totalCycles`)
   - Average number of vector per cycle (`vpc`)

> Note: When analyzing the results in the `stats.txt` file, you will notice that each performance metric appears **three times**. The **first occurrence** of each metric corresponds to the period **during GPU execution of first kernel**, while the **second occurrence** corresponds to the period **during GPU execution of second kernel**. The **third occurrence** is collected **after both GPU kernels have completed**. For the task at hand, you should **focus on the first two sets of statistics**, as they reflect the actual behavior and performance characteristics of the GPU kernels while they are executing.


## Task 3: Analyzing the impact of thread divergence on performance of SpMV kernel(3 points)


The sparse matrix is stored in Compressed Sparse Row (CSR) format, which represents the matrix using three arrays:

- `row_ptr[i]` contains the index into `col_idx` and `data` where row i begins. The number of nonzeros in row i is given by `row_ptr[i+1] - row_ptr[i]`.
- `col_idx[j]` contains the column index of nonzero element j.
- `data[j]` contains the floating-point value of nonzero element j.

A sequential SpMV in CSR format iterates over rows and accumulates the dot product of each row with the vector x:
 
```
for each row i:
    y[i] = 0
    for j from row_ptr[i] to row_ptr[i+1] - 1:
        y[i] += data[j] * x[col_idx[j]]
```

The matrix has 1024 rows and 1024 columns. Row i contains `(i % 64) + 1` nonzeros, so row lengths cycle through 1, 2, 3, ..., 64, 1, 2, 3, ..., 64 across the matrix. This produces a strongly non-uniform distribution of work across rows, with a total of 33,280 nonzero elements and an average of 32.5 nonzeros per row. Kernel and rest of code is provided in the `spmv/` directory and are written in HIP, configured to run within the GEM5 GPU simulation environment.

You are tasked with analyzing and comparing the performance of two kernel implementations:

- **Divergent implementation:** This version launches the SpMV kernel with rows stored in their original order. Each thread in a block processes one row, and a block of 64 threads corresponds to exactly one AMD GCN wavefront.

- **Uniform implementation:** This version sorts the rows of the CSR matrix by their nonzero count in ascending order before launch, then passes the reordered CSR arrays to the same kernel function. 

To build and simulate the kernels, use the provided Makefile in conjunction with the `make_apptainer.sh` script. This script automates the build process inside the pre-configured container environment for GEM5 GPU simulation.

### Experiment Setup:

1. Set the number of compute units to 2, 4, and 8.
2. Run the both of the kernels for each configuration.
3. To compare the performance of the kernels, observe following metrics:
   - Mean and standard deviation of control flow divergence (`controlFlowDivergenceDist::mean` and `controlFlowDivergenceDist::stdev`)
   - Average number of executed vector ALU instructions (`vALUInsts`)
   - Global memory accesses (`globalReads` and `globalWrites`)
   - Coalesced global memory accesses (`coalsrLineAddresses::total`)

> Note: When analyzing the results in the `stats.txt` file, you will notice that each performance metric appears **three times**. The **first occurrence** of each metric corresponds to the period **during GPU  execution of first kernel**, while the **second occurrence** corresponds to the period **during GPU execution of second kernel**. The **third occurrence** is collected **after both GPU kernels have completed**. For the task at hand, you should **focus on the first two sets of statistics**, as they reflect the actual behavior and performance characteristics of the GPU kernels while they are executing.
