#include <hip/hip_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>

#define NUM_BINS 256
#define BLOCK_SIZE 1024


__global__ void histogram_optimized(const unsigned char *image, int *hist, int size) {
    __shared__ int local_hist[NUM_BINS];
    
    // Initialize shared memory
    int t = threadIdx.x;
    if (t < NUM_BINS) local_hist[t] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&local_hist[image[idx]], 1);
    }
    __syncthreads();

    // Merge local histograms into global memory
    if (t < NUM_BINS) {
        atomicAdd(&hist[t], local_hist[t]);
    }
}

int main(int argc, char **argv) {

    int width, height, channels;
    unsigned char *h_img = stbi_load("./histogram/lena_gray.bmp", &width, &height, &channels, 1); // Force grayscale
    

    int img_size = width * height;
    printf("Loaded image %dx%d\n", width, height);

    // Allocate device memory
    unsigned char *d_img;
    int *d_hist;
    hipMalloc(&d_img, img_size);
    hipMalloc(&d_hist, NUM_BINS * sizeof(int));

    // Initialize histogram to zero
    hipMemset(d_hist, 0, NUM_BINS * sizeof(int));

    // Copy image to device
    hipMemcpy(d_img, h_img, img_size, hipMemcpyHostToDevice);

    // Kernel launch parameters
    int threads = BLOCK_SIZE;
    int blocks = (img_size + threads - 1) / threads;


    // --- OPTIMIZED HISTOGRAM ---
    hipMemset(d_hist, 0, NUM_BINS * sizeof(int));
    hipLaunchKernelGGL(histogram_optimized, dim3(blocks), dim3(threads), 0, 0, d_img, d_hist, img_size);
    hipDeviceSynchronize();

    int h_hist_opt[NUM_BINS];
    hipMemcpy(h_hist_opt, d_hist, NUM_BINS * sizeof(int), hipMemcpyDeviceToHost);

    printf("Optimized histogram computed.\n");

    // Compare or print histograms
    for (int i = 0; i < NUM_BINS; ++i) {
        printf("Bin %3d: optimized = %6d\n", i, h_hist_opt[i]);
    }

    // Cleanup
    stbi_image_free(h_img);
    hipFree(d_img);
    hipFree(d_hist);

    return 0;
}
