// reduction_sum.cu
//
// Four parallel-sum reduction kernels, each reducing ARRAYSIZE elements into
// a single float via atomicAdd:
//
//  1. AtomicSumReductionUpSweepKernel
//     Full tree reduction in shared memory. Stride sweeps up: 1 → 2 → 4 → …
//     → blockDim.x/2. At each level threads with tix % (2*stride) == 0 add
//     sdata[tix+stride] into sdata[tix]. One __syncthreads per step. Finally
//     thread 0 atomically adds the block's result to the global output.
//
//  2. AtomicSumReductionDownSweepKernel
//     Full tree reduction in shared memory, but stride sweeps down:
//     blockDim.x/2 → … → 2 → 1. Guard `tix < stride` keeps the partner read
//     tix+stride in bounds; active threads stay contiguous, reducing
//     divergence. One __syncthreads per step. Thread 0 atomically adds the
//     block result.
//
//  3. AtomicSumReductionDownSweepKerneWithWarpPrims
//     Hybrid approach. Shared-memory tree runs down to stride == WARP_SIZE
//     (saving 5 __syncthreads passes vs. kernel 2). The first warp then
//     reduces its 32 partials using __shfl_down_sync register shuffles —
//     no shared memory or barriers for the final steps.
//
//  4. GridStrideReductionKernel
//     Grid-stride loop: each thread processes multiple elements, accumulating
//     a local sum in registers. After the loop, each thread writes its partial
//     to shared memory and a block-level reduction follows the down-sweep +
//     warp-shuffle pattern. CRUCIAL: this kernel must be launched with a SMALL
//     grid sized to the device (numSMs * blocksPerSM), NOT one block per 256
//     elements -- otherwise each thread's loop runs only once and the kernel
//     degenerates into kernel 3 plus loop overhead.
//
// Each kernel is warmed up (untimed) to absorb JIT/context cost, then timed
// over ITERS launches. CUDA events measure pure GPU execution time. Effective
// memory bandwidth (GB/s) is reported alongside, since a reduction is
// memory-bound and bandwidth -- not raw time -- shows how close to the hardware
// limit each kernel runs.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

#define ARRAYSIZE 10000000
#define SEED_NUMBA 42
#define WARP_SIZE 32
#define WARMUP 5  // untimed launches to absorb context/JIT init
#define ITERS 200 // timed launches to average over

// Reduce one warp's 32 values to a single sum, returned in lane 0.
// __shfl_down_sync pulls a value from the lane `stride` higher within the warp,
// entirely in registers: no shared memory, no __syncthreads needed.
__device__ __inline__ float warp_reduce(float val) {
  float partialSum = val;
  for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
    partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
  }
  return partialSum;
}

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

__global__ void
AtomicSumReductionDownSweepKerneWithWarpPrims(const float *input, float *output,
                                              int n) {
  extern __shared__ float sdata[];
  unsigned int tix = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tix;

  // Load with bounds check; padding threads contribute 0.
  sdata[tix] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  // Shared-memory tree, but stop once each warp owns one contiguous chunk:
  // halve stride down to WARP_SIZE (not 1). The remaining 32 partials live in
  // sdata[0..31] and get finished with shuffles below, so we skip the last 5
  // __syncthreads passes.
  for (unsigned int stride = blockDim.x / 2; stride >= WARP_SIZE;
       stride >>= 1) {
    if (tix < stride) {
      sdata[tix] += sdata[tix + stride];
    }
    __syncthreads();
  }

  // The first warp now holds the 32 surviving partials in registers and reduces
  // them to a single value entirely via shuffles -- no shared memory, no sync.
  if (tix < WARP_SIZE) {
    float v = sdata[tix];
    v = warp_reduce(v);
    if (tix == 0) {
      atomicAdd(output, v);
    }
  }
}

__global__ void GridStrideReductionKernel(const float *input, float *output,
                                          int n) {
  extern __shared__ float sdata[];
  unsigned int tix = threadIdx.x;
  unsigned int gridSize = blockDim.x * gridDim.x;

  // Grid-stride loop: each thread accumulates a local sum across multiple
  // elements. With a small device-sized grid, this loop runs many times per
  // thread, amortizing the tree/atomic overhead over far more arithmetic and
  // keeping the loads coalesced (consecutive threads hit consecutive addresses
  // each iteration).
  float localSum = 0.0f;
  for (unsigned int i = blockIdx.x * blockDim.x + tix; i < n; i += gridSize) {
    localSum += input[i];
  }

  // Write the thread's partial sum to shared memory.
  sdata[tix] = localSum;
  __syncthreads();

  // Block-level reduction: down-sweep to WARP_SIZE.
  for (unsigned int stride = blockDim.x / 2; stride >= WARP_SIZE;
       stride >>= 1) {
    if (tix < stride) {
      sdata[tix] += sdata[tix + stride];
    }
    __syncthreads();
  }

  // Final warp-level reduction via shuffles.
  if (tix < WARP_SIZE) {
    float v = sdata[tix];
    v = warp_reduce(v);
    if (tix == 0) {
      atomicAdd(output, v);
    }
  }
}

int main() {
  // 10 million floats; allocate on the heap.
  float *h_inputArray = (float *)malloc(ARRAYSIZE * sizeof(float));
  float *d_inputArray, *d_final_up, *d_final_down, *d_final_warp, *d_final_gs;

  srand(SEED_NUMBA);
  for (int i = 0; i < ARRAYSIZE; i++) {
    h_inputArray[i] = rand() % 10;
  }

  int blockSize = 256;
  int gridSize = (ARRAYSIZE + blockSize - 1) / blockSize;
  size_t shmem = blockSize * sizeof(float);

  // Grid-stride grid is sized to the DEVICE, not the data. A few dozen blocks
  // per SM keeps every SM saturated while letting each thread loop over many
  // elements. This is the configuration that makes the grid-stride loop pay
  // off.
  int numSMs = 0;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  int gsGridSize = numSMs * 32;
  printf(
      "Device SMs: %d   grid-stride grid: %d blocks (vs %d for the rest)\n\n",
      numSMs, gsGridSize, gridSize);

  cudaMalloc(&d_inputArray, ARRAYSIZE * sizeof(float));
  cudaMalloc(&d_final_up, sizeof(float));
  cudaMalloc(&d_final_down, sizeof(float));
  cudaMalloc(&d_final_warp, sizeof(float));
  cudaMalloc(&d_final_gs, sizeof(float));

  cudaMemcpy(d_inputArray, h_inputArray, ARRAYSIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  // CUDA events time the GPU work directly (in milliseconds), independent of
  // host scheduling. One pair is reused for all kernels.
  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  float gpu_ms, total_ms;

  // Bytes read per reduction pass (each element loaded once). Used for the
  // effective-bandwidth figure: GB/s = bytes / seconds.
  double gb = (double)ARRAYSIZE * sizeof(float) / 1e9;

  // Up-sweep kernel
  printf("=== AtomicSumReductionUpSweepKernel ===\n");
  // Warmup: untimed, absorbs context/JIT so it doesn't pollute the average.
  for (int i = 0; i < WARMUP; i++) {
    cudaMemset(d_final_up, 0, sizeof(float));
    AtomicSumReductionUpSweepKernel<<<gridSize, blockSize, shmem>>>(
        d_inputArray, d_final_up, ARRAYSIZE);
  }
  cudaDeviceSynchronize();
  // Timed loop: reset accumulator each iter since atomicAdd builds on contents.
  total_ms = 0.0f;
  for (int i = 0; i < ITERS; i++) {
    cudaMemset(d_final_up, 0, sizeof(float));
    cudaEventRecord(evStart);
    AtomicSumReductionUpSweepKernel<<<gridSize, blockSize, shmem>>>(
        d_inputArray, d_final_up, ARRAYSIZE);
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    cudaEventElapsedTime(&gpu_ms, evStart, evStop);
    total_ms += gpu_ms;
  }
  float t_up = total_ms / ITERS;
  printf("Up-sweep avg time: %.4f ms  (%.1f GB/s)\n", t_up, gb / (t_up / 1e3));

  float gpu_final_up;
  cudaMemcpy(&gpu_final_up, d_final_up, sizeof(float), cudaMemcpyDeviceToHost);

  // Down-sweep kernel
  printf("=== AtomicSumReductionDownSweepKernel ===\n");
  for (int i = 0; i < WARMUP; i++) {
    cudaMemset(d_final_down, 0, sizeof(float));
    AtomicSumReductionDownSweepKernel<<<gridSize, blockSize, shmem>>>(
        d_inputArray, d_final_down, ARRAYSIZE);
  }
  cudaDeviceSynchronize();
  total_ms = 0.0f;
  for (int i = 0; i < ITERS; i++) {
    cudaMemset(d_final_down, 0, sizeof(float));
    cudaEventRecord(evStart);
    AtomicSumReductionDownSweepKernel<<<gridSize, blockSize, shmem>>>(
        d_inputArray, d_final_down, ARRAYSIZE);
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    cudaEventElapsedTime(&gpu_ms, evStart, evStop);
    total_ms += gpu_ms;
  }
  float t_down = total_ms / ITERS;
  printf("Down-sweep avg time: %.4f ms  (%.1f GB/s)\n", t_down,
         gb / (t_down / 1e3));

  float gpu_final_down;
  cudaMemcpy(&gpu_final_down, d_final_down, sizeof(float),
             cudaMemcpyDeviceToHost);

  // Warp-shuffle kernel
  printf("=== AtomicSumReductionDownSweepKerneWithWarpPrims ===\n");
  for (int i = 0; i < WARMUP; i++) {
    cudaMemset(d_final_warp, 0, sizeof(float));
    AtomicSumReductionDownSweepKerneWithWarpPrims<<<gridSize, blockSize,
                                                    shmem>>>(
        d_inputArray, d_final_warp, ARRAYSIZE);
  }
  cudaDeviceSynchronize();
  total_ms = 0.0f;
  for (int i = 0; i < ITERS; i++) {
    cudaMemset(d_final_warp, 0, sizeof(float));
    cudaEventRecord(evStart);
    AtomicSumReductionDownSweepKerneWithWarpPrims<<<gridSize, blockSize,
                                                    shmem>>>(
        d_inputArray, d_final_warp, ARRAYSIZE);
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    cudaEventElapsedTime(&gpu_ms, evStart, evStop);
    total_ms += gpu_ms;
  }
  float t_warp = total_ms / ITERS;
  printf("Warp-prims avg time: %.4f ms  (%.1f GB/s)\n", t_warp,
         gb / (t_warp / 1e3));

  float gpu_final_warp;
  cudaMemcpy(&gpu_final_warp, d_final_warp, sizeof(float),
             cudaMemcpyDeviceToHost);

  // Grid-stride kernel -- launched with the small device-sized grid.
  printf("=== GridStrideReductionKernel ===\n");
  for (int i = 0; i < WARMUP; i++) {
    cudaMemset(d_final_gs, 0, sizeof(float));
    GridStrideReductionKernel<<<gsGridSize, blockSize, shmem>>>(
        d_inputArray, d_final_gs, ARRAYSIZE);
  }
  cudaDeviceSynchronize();
  total_ms = 0.0f;
  for (int i = 0; i < ITERS; i++) {
    cudaMemset(d_final_gs, 0, sizeof(float));
    cudaEventRecord(evStart);
    GridStrideReductionKernel<<<gsGridSize, blockSize, shmem>>>(
        d_inputArray, d_final_gs, ARRAYSIZE);
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    cudaEventElapsedTime(&gpu_ms, evStart, evStop);
    total_ms += gpu_ms;
  }
  float t_gs = total_ms / ITERS;
  printf("Grid-stride avg time: %.4f ms  (%.1f GB/s)\n", t_gs,
         gb / (t_gs / 1e3));

  float gpu_final_gs;
  cudaMemcpy(&gpu_final_gs, d_final_gs, sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);

  // EXACT reference: inputs are whole numbers (rand() % 10), so a long long
  // accumulator sums them with zero rounding. Summing into a float here would
  // itself swamp past 2^24 and give a wrong baseline.
  long long cpu_exact = 0;
  for (int i = 0; i < ARRAYSIZE; i++) {
    cpu_exact += (long long)h_inputArray[i];
  }
  double cpu_final = (double)cpu_exact;

  // RELATIVE tolerance: the GPU result is a float reduction and carries real
  // rounding at this magnitude, so an absolute 1.0 is far too strict. Allow
  // ~0.01% of the total, which comfortably covers float accumulation error
  // while still catching an actual algorithmic bug.
  double tol = cpu_final * 1e-4;

  printf("\nCPU total (exact):    %.0f\n", cpu_final);
  printf("GPU Up-sweep total:   %.0f  (diff %.0f)\n", (double)gpu_final_up,
         (double)gpu_final_up - cpu_final);
  printf("GPU Down-sweep total: %.0f  (diff %.0f)\n", (double)gpu_final_down,
         (double)gpu_final_down - cpu_final);
  printf("GPU Warp-prims total: %.0f  (diff %.0f)\n", (double)gpu_final_warp,
         (double)gpu_final_warp - cpu_final);
  printf("GPU Grid-stride total:%.0f  (diff %.0f)\n", (double)gpu_final_gs,
         (double)gpu_final_gs - cpu_final);
  printf("Tolerance (+/-):      %.0f\n", tol);

  bool up_ok = fabs((double)gpu_final_up - cpu_final) < tol;
  bool down_ok = fabs((double)gpu_final_down - cpu_final) < tol;
  bool warp_ok = fabs((double)gpu_final_warp - cpu_final) < tol;
  bool gs_ok = fabs((double)gpu_final_gs - cpu_final) < tol;

  printf("Up-sweep PASSED: %s\n", up_ok ? "True" : "False");
  printf("Down-sweep PASSED: %s\n", down_ok ? "True" : "False");
  printf("Warp-prims PASSED: %s\n", warp_ok ? "True" : "False");
  printf("Grid-stride PASSED: %s\n", gs_ok ? "True" : "False");

  // Relative speeds, using the slowest kernel as the baseline.
  printf("Down-sweep speedup vs up-sweep: %.2fx\n", t_up / t_down);
  printf("Warp-prims speedup vs up-sweep: %.2fx\n", t_up / t_warp);
  printf("Grid-stride speedup vs up-sweep: %.2fx\n", t_up / t_gs);

  cudaFree(d_inputArray);
  cudaFree(d_final_up);
  cudaFree(d_final_down);
  cudaFree(d_final_warp);
  cudaFree(d_final_gs);
  free(h_inputArray);

  // All kernels must match the exact reference within tolerance to pass.
  bool ok = up_ok && down_ok && warp_ok && gs_ok;
  return ok ? 0 : 1;
}
