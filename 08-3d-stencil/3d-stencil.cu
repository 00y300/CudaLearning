// ch8_3d_stencil.cu — 7-point 3D stencil with tiled shared memory
//
// Build: cmake -B build -S . && cmake --build build --target 08-3d-stencil
// Run:   ./build/08-3d-stencil
//
// Creates a synthetic 3D volume (256^3), applies one iteration of the
// 7-point Laplacian stencil on GPU using tiled shared memory with
// register caching and halo regions.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Grid and tile dimensions
#define GRID_D 256
#define GRID_H 256
#define GRID_W 256
#define TILE_SIZE 16
#define HALO (1)
#define SHMEM_D (TILE_SIZE + 2 * HALO)
#define SHMEM_H (TILE_SIZE + 2 * HALO)
#define SHMEM_W (TILE_SIZE + 2 * HALO)

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while(0)

// 7-point stencil:
//   out[i,j,k] = (in[i-1,j,k] + in[i+1,j,k] +
//                 in[i,j-1,k] + in[i,j+1,k] +
//                 in[i,j,k-1] + in[i,j,k+1] -
//                 6.0f * in[i,j,k]) / 24.0f
__global__ void stencil_3d(const float* in, float* out, int D, int H, int W) {
    __shared__ float tile[SHMEM_D][SHMEM_H][SHMEM_W];

    // Output voxel coordinates
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int dep = blockIdx.z * TILE_SIZE + threadIdx.z;

    // Boundary check for output
    if (col >= W || row >= H || dep >= D) return;

    // Load input tile into shared memory including 1-voxel halo on all 6 faces
    for (int k = threadIdx.z; k < SHMEM_D; k += blockDim.z)
        for (int j = threadIdx.y; j < SHMEM_H; j += blockDim.y)
            for (int i = threadIdx.x; i < SHMEM_W; i += blockDim.x) {
                int gC = blockIdx.x * TILE_SIZE + i - HALO;
                int gR = blockIdx.y * TILE_SIZE + j - HALO;
                int gD = blockIdx.z * TILE_SIZE + k - HALO;
                tile[k][j][i] = (gC >= 0 && gC < W &&
                                 gR >= 0 && gR < H &&
                                 gD >= 0 && gD < D)
                                ? in[gD * W * H + gR * W + gC]
                                : 0.0f;
            }

    __syncthreads();

    // Only threads in the interior (non-halo) compute the stencil
    if (threadIdx.x == 0 || threadIdx.x == TILE_SIZE - 1 ||
        threadIdx.y == 0 || threadIdx.y == TILE_SIZE - 1 ||
        threadIdx.z == 0 || threadIdx.z == TILE_SIZE - 1)
        return;

    // Register-cached center value
    float center = tile[threadIdx.z + HALO][threadIdx.y + HALO][threadIdx.x + HALO];

    // 7-point stencil with register caching of neighbors
    float left   = tile[threadIdx.z + HALO][threadIdx.y + HALO][threadIdx.x + HALO - 1];
    float right  = tile[threadIdx.z + HALO][threadIdx.y + HALO][threadIdx.x + HALO + 1];
    float bottom = tile[threadIdx.z + HALO][threadIdx.y + HALO - 1][threadIdx.x + HALO];
    float top    = tile[threadIdx.z + HALO][threadIdx.y + HALO + 1][threadIdx.x + HALO];
    float back   = tile[threadIdx.z + HALO - 1][threadIdx.y + HALO][threadIdx.x + HALO];
    float front  = tile[threadIdx.z + HALO + 1][threadIdx.y + HALO][threadIdx.x + HALO];

    out[dep * W * H + row * W + col] = (left + right + bottom + top + back + front - 6.0f * center) / 24.0f;
}

int main() {
    int N = GRID_D * GRID_H * GRID_W;
    printf("3D Stencil: %d^3 volume (%.0fM voxels)\n", GRID_D, N / 1e6);

    // Host arrays
    float* h_in  = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));

    // Seed with a smooth pattern (sine wave)
    for (int k = 0; k < GRID_D; k++)
        for (int j = 0; j < GRID_H; j++)
            for (int i = 0; i < GRID_W; i++)
                h_in[k * GRID_W * GRID_H + j * GRID_W + i] =
                    sinf(2.0f * 3.14159f * (i + j + k) / GRID_W) * 0.5f + 0.5f;

    // Device buffers
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Upload input
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch config
    dim3 block(TILE_SIZE, TILE_SIZE, TILE_SIZE);
    dim3 grid((GRID_W + TILE_SIZE - 1) / TILE_SIZE,
              (GRID_H + TILE_SIZE - 1) / TILE_SIZE,
              (GRID_D + TILE_SIZE - 1) / TILE_SIZE);

    float ms = 0;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    CUDA_CHECK(cudaEventRecord(t0));

    stencil_3d<<<grid, block>>>(d_in, d_out, GRID_D, GRID_H, GRID_W);

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // Download result
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("GPU 3D stencil: %.2f ms (%.2f TFLOP/s estimate)\n",
           ms, (N * 29.0f / 1e12) / (ms / 1000.0f));

    // Spot-check: verify a known interior point
    int midD = GRID_D / 2, midH = GRID_H / 2, midW = GRID_W / 2;
    printf("Sample output[%d][%d][%d] = %.6f\n",
           midD, midH, midW, h_out[midD * GRID_W * GRID_H + midH * GRID_W + midW]);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    printf("Done. Hardware: RTX 5090 (Blackwell, sm_120)\n");
    return 0;
}
