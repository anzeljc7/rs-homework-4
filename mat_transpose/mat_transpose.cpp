/*
 * transpose_benchmark.hip
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Configuration parameters
#define N        512        /* matrix dimension: N x N elements        */
#define TILE     16          /* tile / block side length                 */
#define TILE_PAD (TILE + 1)  /* LDS row width (+1 avoids bank conflicts) */


#define HIP_CHECK(cmd)                                                    \
    do {                                                                  \
        hipError_t _e = (cmd);                                            \
        if (_e != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d  \"%s\"\n",              \
                    __FILE__, __LINE__, hipGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

///  KERNEL 1 – Naive transpose          
__global__ void matrix_transpose_naive(const float * __restrict__ A,
                                              float * __restrict__ B,
                                              int n)
{
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    if (row < n && col < n)
        B[col * n + row] = A[row * n + col];
}

//  KERNEL 2 – Tiled transpose via LDS  (GOOD: both accesses coalesced)
__global__ void matrix_transpose_shared(const float * __restrict__ A,
                                               float * __restrict__ B,
                                               int n)
{
    /* LDS tile: TILE rows x TILE_PAD columns (padding avoids bank conflicts) */
    __shared__ float tile[TILE][TILE_PAD];

    /* --- Phase 1: coalesced load from global memory into LDS --- */
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    if (row < n && col < n)
        tile[threadIdx.y][threadIdx.x] = A[row * n + col];

    __syncthreads();

    /* --- Phase 2: coalesced store from LDS to global memory --- */
    /* Re-map threads so threadIdx.x is the fast (column) index in output */
    int out_col = blockIdx.y * TILE + threadIdx.x;
    int out_row = blockIdx.x * TILE + threadIdx.y;

    if (out_row < n && out_col < n)
        B[out_row * n + out_col] = tile[threadIdx.x][threadIdx.y];
}

// Host helper functions
static void fill_matrix(float *M, int n)
{
    for (int i = 0; i < n * n; i++)
        M[i] = (float)i;
}

static int verify_transpose(const float *A, const float *B, int n)
{
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            if (fabsf(B[c * n + r] - A[r * n + c]) > 1e-5f) {
                fprintf(stderr,
                        "Verify FAILED: B[%d][%d] = %.1f, expected %.1f\n",
                        c, r, B[c * n + r], A[r * n + c]);
                return 0;
            }
    return 1;
}

// Main function
int main(void)
{
    const int    n     = N;
    const size_t bytes = (size_t)n * n * sizeof(float);

    /* ---- host buffers ---- */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    if (!h_A || !h_B) { perror("malloc"); exit(EXIT_FAILURE); }

    fill_matrix(h_A, n);

    /* ---- device buffers ---- */
    float *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));

    /* ---- launch configuration ---- */
    dim3 block(TILE, TILE);
    dim3 grid(n / TILE, n / TILE);

    //  Kernel 1: naive transpose (uncoalesced writes)                
    printf("[Kernel 1] matrix_transpose_naive\n");

    HIP_CHECK(hipMemset(d_B, 0, bytes));
    hipLaunchKernelGGL(matrix_transpose_naive, grid, block, 0, 0,
                       d_A, d_B, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_B, d_B, bytes, hipMemcpyDeviceToHost));
    printf("           result : %s\n\n",
           verify_transpose(h_A, h_B, n) ? "CORRECT" : "WRONG");

    //  Kernel 2: tiled transpose via shared memory (both coalesced)   
    HIP_CHECK(hipMemset(d_B, 0, bytes));
    hipLaunchKernelGGL(matrix_transpose_shared, grid, block, 0, 0,
                       d_A, d_B, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_B, d_B, bytes, hipMemcpyDeviceToHost));
    printf("           result : %s\n\n",
           verify_transpose(h_A, h_B, n) ? "CORRECT" : "WRONG");

    /* ---- cleanup ---- */
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    free(h_A);
    free(h_B);

    printf("Done. Check stats.txt for memory access statistics.\n");
    return 0;
}
