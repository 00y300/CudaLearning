// reduction_sum.cu
//
// Two kernels for parallel sum reduction, both using atomicAdd to accumulate
// per-block partial sums into a global result:
//
//  1. AtomicSumReductionUpSweepKernel  (up-sweep / unfold-up)
//     Stride iterates forward: 1 -> 2 -> 4 -> ... -> blockDim.x/2.
//     Threads with tix % (2*stride) == 0 perform the partial-sum.
//
//  2. AtomicSumReductionDownSweepKernel (down-sweep / unfold-down)
//     Stride iterates backward: blockDim.x/2 -> ... -> 2 -> 1.
//     Threads with tix < stride perform the partial-sum.
//
// Both kernels load input into shared memory, tree-reduce within each block,
// then use a single atomicAdd per block to fold into the global accumulator.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

#define ARRAYSIZE 125000
#define SEEDNUMBA 42

__global__ void AtomicSumReductionUpSweepKernel(const float *input,
                                                float *output, int n) {
  extern __shared__ float sdata[];
  unsigned int tix = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tix;

  // Load with bounds check; padding threads contribute 0.
  sdata[tix] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  // Tree reduction within the block.
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tix % (2 * stride) == 0) {
      sdata[tix] += sdata[tix + stride];
    }
    __syncthreads();
  }

  // One atomic per block accumulates the partial into the final result.
  if (tix == 0) {
    atomicAdd(output, sdata[0]);
  }
}

__global__ void AtomicSumReductionDownSweepKernel(const float *input,
                                                  float *output, int n) {
  extern __shared__ float sdata[];
  unsigned int tix = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tix;

  // Load with bounds check; padding threads contribute 0.
  sdata[tix] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  // Tree reduction within the block (sequential addressing).
  // stride: blockDim.x/2 -> ... -> 1. Guard `tix < stride` keeps the partner
  // index `tix + stride` strictly less than blockDim.x, so the read is always
  // in bounds, and active threads remain contiguous to minimize divergence.
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tix < stride) {
      sdata[tix] += sdata[tix + stride];
    }
    __syncthreads();
  }

  // One atomic per block accumulates the partial into the final result.
  if (tix == 0) {
    atomicAdd(output, sdata[0]);
  }
}

int main() {
  // 125000 floats is too large for the stack; allocate on the heap.
  float *h_inputArray = (float *)malloc(ARRAYSIZE * sizeof(float));
  float *d_inputArray, *d_final_up, *d_final_down;

  srand(SEEDNUMBA);
  for (int i = 0; i < ARRAYSIZE; i++) {
    h_inputArray[i] = rand() % 10;
  }

  int blockSize = 256;
  int gridSize = (ARRAYSIZE + blockSize - 1) / blockSize;

  cudaMalloc(&d_inputArray, ARRAYSIZE * sizeof(float));
  cudaMalloc(&d_final_up, sizeof(float));
  cudaMalloc(&d_final_down, sizeof(float));

  cudaMemcpy(d_inputArray, h_inputArray, ARRAYSIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  // CUDA events time the GPU work directly (in milliseconds), independent of
  // host scheduling. One pair is reused for both kernels.
  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  float gpu_ms;

  // Up-sweep kernel
  printf("=== AtomicSumReductionUpSweepKernel ===\n");
  cudaMemset(d_final_up, 0, sizeof(float));

  cudaEventRecord(evStart);
  AtomicSumReductionUpSweepKernel<<<gridSize, blockSize,
                                    blockSize * sizeof(float)>>>(
      d_inputArray, d_final_up, ARRAYSIZE);
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&gpu_ms, evStart, evStop);
  printf("Up-sweep time: %.4f ms\n", gpu_ms);

  float gpu_final_up;
  cudaMemcpy(&gpu_final_up, d_final_up, sizeof(float), cudaMemcpyDeviceToHost);

  // Down-sweep kernel
  printf("=== AtomicSumReductionDownSweepKernel ===\n");
  cudaMemset(d_final_down, 0, sizeof(float));

  cudaEventRecord(evStart);
  AtomicSumReductionDownSweepKernel<<<gridSize, blockSize,
                                      blockSize * sizeof(float)>>>(
      d_inputArray, d_final_down, ARRAYSIZE);
  cudaEventRecord(evStop);
  cudaEventSynchronize(evStop);
  cudaEventElapsedTime(&gpu_ms, evStart, evStop);
  printf("Down-sweep time: %.4f ms\n", gpu_ms);

  float gpu_final_down;
  cudaMemcpy(&gpu_final_down, d_final_down, sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);

  float cpu_final = 0.0f;
  for (int i = 0; i < ARRAYSIZE; i++) {
    cpu_final += h_inputArray[i];
  }

  printf("CPU total: %.2f\n", cpu_final);
  printf("GPU Up-sweep total: %.2f\n", gpu_final_up);
  printf("GPU Down-sweep total: %.2f\n", gpu_final_down);
  printf("Up-sweep PASSED: %s\n",
         fabs(gpu_final_up - cpu_final) < 1.0f ? "True" : "False");
  printf("Down-sweep PASSED: %s\n",
         fabs(gpu_final_down - cpu_final) < 1.0f ? "True" : "False");

  cudaFree(d_inputArray);
  cudaFree(d_final_up);
  cudaFree(d_final_down);
  free(h_inputArray);

  // Both kernels must match the CPU reference for the program to pass.
  bool ok = fabs(gpu_final_up - cpu_final) < 1.0f &&
            fabs(gpu_final_down - cpu_final) < 1.0f;
  return ok ? 0 : 1;
}
