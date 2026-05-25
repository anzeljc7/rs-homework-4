/*
 * spmv_benchmark.hip
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Configuration parameters 
#define NUM_ROWS    1024          /* matrix rows (and columns)          */
#define WARP_SIZE   64            /* AMD GCN wavefront width            */
#define BLOCK_SIZE  WARP_SIZE     /* one wavefront per block            */


#define HIP_CHECK(cmd)                                                    \
    do {                                                                  \
        hipError_t _e = (cmd);                                            \
        if (_e != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d  \"%s\"\n",              \
                    __FILE__, __LINE__, hipGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// SpMV KERNEL – one thread per row  (used for BOTH launches)        
__global__ void spmv_csr(const int   * __restrict__ row_ptr,
                         const int   * __restrict__ col_idx,
                         const float * __restrict__ data,
                         const float * __restrict__ x,
                               float * __restrict__ y,
                         int num_rows)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid >= num_rows) return;

    float sum = 0.0f;
    int   row_start = row_ptr[gid];
    int   row_end   = row_ptr[gid + 1];

    for (int j = row_start; j < row_end; j++)
        sum += data[j] * x[col_idx[j]];

    y[gid] = sum;
}

//  Host: generate sparse matrix in CSR                                 */
/*
 * Row i has (i % WARP_SIZE + 1) non-zeros.
 * This creates a strongly non-uniform pattern: lengths cycle 1,2,...,64
 */
static void generate_csr(int nrows, int ncols,
                         int **row_ptr_out, int **col_idx_out,
                         float **data_out,  int *nnz_out)
{
    int *row_ptr = (int *)malloc((nrows + 1) * sizeof(int));
    row_ptr[0] = 0;
    for (int i = 0; i < nrows; i++)
        row_ptr[i + 1] = row_ptr[i] + (i % WARP_SIZE + 1);
    int nnz = row_ptr[nrows];

    int   *col_idx = (int *)  malloc(nnz * sizeof(int));
    float *data    = (float *)malloc(nnz * sizeof(float));

    srand(42);
    for (int i = 0; i < nrows; i++) {
        int start = row_ptr[i];
        int len   = row_ptr[i + 1] - row_ptr[i];
        for (int k = 0; k < len; k++) {
            col_idx[start + k] = rand() % ncols;
            data   [start + k] = (float)(rand() % 100) / 100.0f + 0.01f;
        }
    }

    *row_ptr_out = row_ptr;
    *col_idx_out = col_idx;
    *data_out    = data;
    *nnz_out     = nnz;
}

//  Host: sort rows by non-zero count, rebuild CSR                      
static void sort_rows_by_nnz(int nrows,
                              const int *row_ptr, const int *col_idx,
                              const float *data,
                              int **s_row_ptr_out, int **s_col_idx_out,
                              float **s_data_out,  int *perm_out)
{
    /* build permutation sorted by row length ascending */
    int *perm = (int *)malloc(nrows * sizeof(int));
    for (int i = 0; i < nrows; i++) perm[i] = i;

    /* simple insertion sort — nrows=1024 is small enough */
    for (int i = 1; i < nrows; i++) {
        int key = perm[i];
        int key_len = row_ptr[key + 1] - row_ptr[key];
        int j = i - 1;
        while (j >= 0 && (row_ptr[perm[j]+1]-row_ptr[perm[j]]) > key_len) {
            perm[j + 1] = perm[j];
            j--;
        }
        perm[j + 1] = key;
    }

    /* rebuild CSR in sorted order */
    int *s_row_ptr = (int *)malloc((nrows + 1) * sizeof(int));
    s_row_ptr[0] = 0;
    for (int i = 0; i < nrows; i++) {
        int orig_row = perm[i];
        int len      = row_ptr[orig_row + 1] - row_ptr[orig_row];
        s_row_ptr[i + 1] = s_row_ptr[i] + len;
    }
    int nnz = s_row_ptr[nrows];

    int   *s_col_idx = (int *)  malloc(nnz * sizeof(int));
    float *s_data    = (float *)malloc(nnz * sizeof(float));

    for (int i = 0; i < nrows; i++) {
        int orig_row  = perm[i];
        int orig_start = row_ptr[orig_row];
        int len        = row_ptr[orig_row + 1] - row_ptr[orig_row];
        memcpy(s_col_idx + s_row_ptr[i], col_idx + orig_start,
               len * sizeof(int));
        memcpy(s_data    + s_row_ptr[i], data    + orig_start,
               len * sizeof(float));
    }

    *s_row_ptr_out  = s_row_ptr;
    *s_col_idx_out  = s_col_idx;
    *s_data_out     = s_data;
    memcpy(perm_out, perm, nrows * sizeof(int));
    free(perm);
}

// Host: verify results                                               
static int verify(const float *y_div, const float *y_sorted,
                  const int *perm, int nrows)
{
    for (int i = 0; i < nrows; i++) {
        float expected = y_div[perm[i]];
        float got      = y_sorted[i];
        if (fabsf(expected - got) > 1e-3f * (fabsf(expected) + 1e-6f)) {
            fprintf(stderr,
                    "Verify FAILED: sorted row %d (orig %d): "
                    "y_div=%.6f  y_sorted=%.6f\n",
                    i, perm[i], expected, got);
            return 0;
        }
    }
    return 1;
}

// main                                                                
int main(void)
{
    const int nrows = NUM_ROWS;
    const int ncols = NUM_ROWS;

    int *h_row_ptr, *h_col_idx; float *h_data; int nnz;
    generate_csr(nrows, ncols, &h_row_ptr, &h_col_idx, &h_data, &nnz);

    int *h_s_row_ptr, *h_s_col_idx; float *h_s_data;
    int *perm = (int *)malloc(nrows * sizeof(int));
    sort_rows_by_nnz(nrows, h_row_ptr, h_col_idx, h_data,
                     &h_s_row_ptr, &h_s_col_idx, &h_s_data, perm);

    float *h_x = (float *)malloc(ncols * sizeof(float));
    for (int i = 0; i < ncols; i++) h_x[i] = (float)i / ncols + 0.01f;

    float *h_y_div    = (float *)calloc(nrows, sizeof(float));
    float *h_y_sorted = (float *)calloc(nrows, sizeof(float));


    int   *d_row_ptr,   *d_col_idx;
    int   *d_s_row_ptr, *d_s_col_idx;
    float *d_data,      *d_s_data;
    float *d_x, *d_y;

    HIP_CHECK(hipMalloc(&d_row_ptr,   (nrows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_idx,   nnz         * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_data,      nnz         * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_s_row_ptr, (nrows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_s_col_idx, nnz         * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_s_data,    nnz         * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x,         ncols       * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y,         nrows       * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_row_ptr,   h_row_ptr,   (nrows+1)*sizeof(int),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_idx,   h_col_idx,   nnz*sizeof(int),         hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_data,      h_data,      nnz*sizeof(float),       hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_s_row_ptr, h_s_row_ptr, (nrows+1)*sizeof(int),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_s_col_idx, h_s_col_idx, nnz*sizeof(int),         hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_s_data,    h_s_data,    nnz*sizeof(float),       hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,         h_x,         ncols*sizeof(float),     hipMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(nrows / BLOCK_SIZE);


    // CSR kernel with original row order (divergent)
    HIP_CHECK(hipMemset(d_y, 0, nrows * sizeof(float)));
    hipLaunchKernelGGL(spmv_csr, grid, block, 0, 0,
                       d_row_ptr, d_col_idx, d_data, d_x, d_y, nrows);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_y_div, d_y, nrows * sizeof(float),
                        hipMemcpyDeviceToHost));
    printf("           done.\n\n");

    // CSR kernel with sorted rows (uniform)
    HIP_CHECK(hipMemset(d_y, 0, nrows * sizeof(float)));
    hipLaunchKernelGGL(spmv_csr, grid, block, 0, 0,
                       d_s_row_ptr, d_s_col_idx, d_s_data, d_x, d_y, nrows);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_y_sorted, d_y, nrows * sizeof(float),
                        hipMemcpyDeviceToHost));
    printf("           done.\n\n");

    /* ---- verify ---- */
    printf("Cross-verification: %s\n",
           verify(h_y_div, h_y_sorted, perm, nrows)
               ? "PASSED (results match)"
               : "FAILED (results differ!)");

    /* ---- cleanup ---- */
    HIP_CHECK(hipFree(d_row_ptr));   HIP_CHECK(hipFree(d_col_idx));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_s_row_ptr)); HIP_CHECK(hipFree(d_s_col_idx));
    HIP_CHECK(hipFree(d_s_data));
    HIP_CHECK(hipFree(d_x));         HIP_CHECK(hipFree(d_y));

    free(h_row_ptr);   free(h_col_idx);   free(h_data);
    free(h_s_row_ptr); free(h_s_col_idx); free(h_s_data);
    free(h_x); free(h_y_div); free(h_y_sorted); free(perm);

    return 0;
}
