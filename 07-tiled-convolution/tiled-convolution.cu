// ch7_blur.cu — load image, apply Gaussian blur on GPU, save PNG
//
// Build: cmake -B build -S . && cmake --build build --target 07-tiled-convolution
// Run:   ./build/07-tiled-convolution assets/f1Car.jpg assets/output_blur.png
//
// Image credit: f1Car.jpg from Unsplash (https://unsplash.com/photos/a-race-car-is-driving-on-the-track-W0haDrVTW58)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 15x15 Gaussian blur mask in fast constant memory
#define MASK_H 15
#define MASK_W 15
__constant__ float d_mask[MASK_H * MASK_W];

#define TILE_SIZE 32
#define SHMEM_H (TILE_SIZE + MASK_H - 1)
#define SHMEM_W (TILE_SIZE + MASK_W - 1)

#define BLUR_PASSES 5   // number of times to apply blur (more = stronger effect)

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while(0)

__global__ void conv2d_tiled(const float* in, float* out, int H, int W) {
    __shared__ float tile[SHMEM_H][SHMEM_W];

    int rH = MASK_H / 2, rW = MASK_W / 2;
    int originR = blockIdx.y * TILE_SIZE - rH;
    int originC = blockIdx.x * TILE_SIZE - rW;

    for (int i = threadIdx.y; i < SHMEM_H; i += blockDim.y)
        for (int j = threadIdx.x; j < SHMEM_W; j += blockDim.x) {
            int gR = originR + i, gC = originC + j;
            tile[i][j] = (gR >= 0 && gR < H && gC >= 0 && gC < W)
                         ? in[gR * W + gC] : 0.f;
        }
    __syncthreads();

    int outR = blockIdx.y * TILE_SIZE + threadIdx.y;
    int outC = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (outR >= H || outC >= W) return;

    float sum = 0.f;
    for (int mR = 0; mR < MASK_H; ++mR)
        for (int mC = 0; mC < MASK_W; ++mC)
            sum += tile[threadIdx.y + mR][threadIdx.x + mC]
                 * d_mask[mR * MASK_W + mC];

    out[outR * W + outC] = sum;
}

int main(int argc, char** argv) {
    const char* in_path  = (argc > 1) ? argv[1] : "f1Car.jpg";
    const char* out_path = (argc > 2) ? argv[2] : "output_blur.png";

    // Load image (force RGB)
    int W, H, ch;
    unsigned char* img = stbi_load(in_path, &W, &H, &ch, 3);
    if (!img) { fprintf(stderr, "Failed to load %s\n", in_path); return 1; }
    printf("Loaded: %s (%dx%d)\n", in_path, W, H);

    int N = H * W;

    // Separate into R, G, B float planes [0,1]
    float *h_r = (float*)malloc(N*sizeof(float));
    float *h_g = (float*)malloc(N*sizeof(float));
    float *h_b = (float*)malloc(N*sizeof(float));
    for (int i = 0; i < N; i++) {
        h_r[i] = img[i*3+0] / 255.f;
        h_g[i] = img[i*3+1] / 255.f;
        h_b[i] = img[i*3+2] / 255.f;
    }
    stbi_image_free(img);

    // Compute Gaussian kernel at runtime (sigma=3, 15x15)
    float h_blur[MASK_H * MASK_W];
    float sigma = 3.0f;
    float sum = 0;
    for (int i = 0; i < MASK_H; i++)
        for (int j = 0; j < MASK_W; j++) {
            float dx = i - MASK_H / 2.0f, dy = j - MASK_W / 2.0f;
            h_blur[i * MASK_W + j] = expf(-(dx*dx + dy*dy) / (2*sigma*sigma));
            sum += h_blur[i * MASK_W + j];
        }
    for (int i = 0; i < MASK_H * MASK_W; i++) h_blur[i] /= sum;
    CUDA_CHECK(cudaMemcpyToSymbol(d_mask, h_blur, sizeof(h_blur)));

    // GPU buffers — two buffers so we can ping-pong between passes
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(float)));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((W+TILE_SIZE-1)/TILE_SIZE, (H+TILE_SIZE-1)/TILE_SIZE);

    float *out_r = (float*)malloc(N*sizeof(float));
    float *out_g = (float*)malloc(N*sizeof(float));
    float *out_b = (float*)malloc(N*sizeof(float));
    float* dst[] = {out_r, out_g, out_b};
    float* src[] = {h_r,   h_g,   h_b};

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    CUDA_CHECK(cudaEventRecord(t0));

    for (int c = 0; c < 3; c++) {
        // Upload channel into buffer A
        CUDA_CHECK(cudaMemcpy(d_a, src[c], N*sizeof(float), cudaMemcpyHostToDevice));

 
        float *in = d_a, *out = d_b;
        for (int p = 0; p < BLUR_PASSES; p++) {
            conv2d_tiled<<<grid, block>>>(in, out, H, W);
            float* tmp = in; in = out; out = tmp;   // swap
        }
        // After an odd number of passes 'in' points to the final result
        CUDA_CHECK(cudaMemcpy(dst[c], in, N*sizeof(float), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    printf("GPU blur: %.2f ms  (%d passes, 15x15 kernel)\n", ms, BLUR_PASSES);

    // Pack back to RGB uint8
    unsigned char* out_img = (unsigned char*)malloc(N*3);
    for (int i = 0; i < N; i++) {
        auto clamp = [](float v){ return (unsigned char)(v<0?0:v>1?255:(int)(v*255)); };
        out_img[i*3+0] = clamp(out_r[i]);
        out_img[i*3+1] = clamp(out_g[i]);
        out_img[i*3+2] = clamp(out_b[i]);
    }

    stbi_write_png(out_path, W, H, 3, out_img, W*3);
    printf("Saved:  %s\n", out_path);

    cudaFree(d_a); cudaFree(d_b);
    free(h_r); free(h_g); free(h_b);
    free(out_r); free(out_g); free(out_b);
    free(out_img);
    return 0;
}
