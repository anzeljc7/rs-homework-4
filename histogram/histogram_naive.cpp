#include <hip/hip_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>

#define NUM_BINS 256
#define BLOCK_SIZE 1024

__global__ void histogram_naive(const unsigned char *image, int *hist, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&hist[image[idx]], 1);
    }
}


int main(int argc, char **argv) {
 

    int width, height, channels;
    unsigned char *h_img = stbi_load("./histogram/lena_gray.bmp", &width, &height, &channels, 1); // Force grayscale

    if (!h_img) {
        printf("Failed to load image: %s\n", argv[1]);
        return 1;
    }

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

    // --- NAIVE HISTOGRAM ---
    hipMemset(d_hist, 0, NUM_BINS * sizeof(int));
    hipLaunchKernelGGL(histogram_naive, dim3(blocks), dim3(threads), 0, 0, d_img, d_hist, img_size);
    hipDeviceSynchronize();

    int h_hist_naive[NUM_BINS];
    hipMemcpy(h_hist_naive, d_hist, NUM_BINS * sizeof(int), hipMemcpyDeviceToHost);

    printf("Naive histogram computed.\n");



    // Compare or print histograms
    for (int i = 0; i < NUM_BINS; ++i) {
        printf("Bin %3d: naive = %6d \n", i, h_hist_naive[i]);
    }

    // Cleanup
    stbi_image_free(h_img);
    hipFree(d_img);
    hipFree(d_hist);

    return 0;
}
