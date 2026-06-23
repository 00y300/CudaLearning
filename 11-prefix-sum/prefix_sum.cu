// global_scan_5090.cu
// Full GLOBAL inclusive scan (prefix sum) of ~1 billion elements on the
// NVIDIA GeForce RTX 5090 (Blackwell, compute capability sm_120).
//
// A single block can only scan its own COARSE_FACTOR*BLOCK_DIM tile, so a
// billion-element scan uses the classic THREE-PASS hierarchical approach:
//
//   PASS 1  scan_kernel        : each block inclusively scans its tile and
//                                writes that tile's grand total to blockSums[].
//   PASS 2  scan_block_sums    : inclusively scan the (relatively tiny)
//                                blockSums[] array so each entry becomes the
//                                running total up to and including that block.
//   PASS 3  add_block_offsets  : add the *exclusive* prefix of each block
//                                (blockSums[blockIdx-1]) to every element of
//                                that block, stitching the tiles into one
//                                globally correct inclusive scan.

#include <cstdio>
#include <cstdlib>
#include <vector>

// ---- Hardware / tiling parameters ----
#define WARP_SIZE 32                      // physical warp width on the 5090
#define BLOCK_DIM 256                     // threads per block (8 warps)
#define COARSE_FACTOR 16                  // elements each thread folds first
#define NUM_WARPS (BLOCK_DIM / WARP_SIZE) // = 8 warps per block
#define FULL_MASK 0xffffffffu             // all 32 lanes active
#define TILE (COARSE_FACTOR * BLOCK_DIM)  // 4096 elements per block

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__device__ inline unsigned laneIdx() { return threadIdx.x & (WARP_SIZE - 1); }
__device__ inline unsigned warpIdx() { return threadIdx.x >> 5; }

// ====================================================================
// WARP-LEVEL inclusive scan (register shuffles, no shared memory).
// log2(32)=5 steps; each lane ends with the sum of itself + earlier lanes.
// ====================================================================
__device__ inline float warpScan(float val) {
  for (unsigned stride = 1; stride < WARP_SIZE; stride *= 2) {
    float leftVal = __shfl_up_sync(FULL_MASK, val, stride);
    if (laneIdx() >= stride)
      val += leftVal;
  }
  return val;
}

// ====================================================================
// BLOCK-LEVEL scan via scan / scan / add (Fig 11.9).
//   1) scan each warp,  2) scan the 8 warp totals with warp 0,
//   3) add each warp the running total of all warps before it.
// ====================================================================
__device__ inline float blockScan(float val) {
  val = warpScan(val); // phase 1: per-warp scan

  __shared__ float warpSums_s[NUM_WARPS];
  if (laneIdx() == WARP_SIZE - 1)
    warpSums_s[warpIdx()] = val; // last lane = warp total
  __syncthreads();

  if (warpIdx() == 0) { // phase 2: scan the warp totals
    float warpSum = (threadIdx.x < NUM_WARPS) ? warpSums_s[threadIdx.x] : 0.0f;
    warpSum = warpScan(warpSum);
    if (threadIdx.x < NUM_WARPS)
      warpSums_s[threadIdx.x] = warpSum;
  }
  __syncthreads();

  if (warpIdx() > 0)
    val += warpSums_s[warpIdx() - 1]; // phase 3: add earlier warps
  return val;
}

// ====================================================================
// PASS 1 — coarsened per-block inclusive scan (Fig 11.12).
// Each block scans its TILE and, if blockSums != nullptr, records the
// tile's grand total so later passes can offset the tiles correctly.
// ====================================================================
__global__ void scan_kernel(const float *input, float *output, float *blockSums,
                            unsigned int N) {
  unsigned int blockSegment = (unsigned long long)blockIdx.x * TILE;

  // A) Coalesced, strided load into shared memory.
  __shared__ float buffer_s[TILE];
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    unsigned long long g =
        (unsigned long long)blockSegment + c * BLOCK_DIM + threadIdx.x;
    buffer_s[c * BLOCK_DIM + threadIdx.x] = (g < N) ? input[g] : 0.0f;
  }
  __syncthreads();

  // B) Each thread serially folds its COARSE_FACTOR private elements.
  unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
  for (unsigned c = 1; c < COARSE_FACTOR; ++c)
    buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];

  // C) Block-wide scan of the per-thread totals.
  float threadSum = buffer_s[threadSegment + COARSE_FACTOR - 1];
  threadSum = blockScan(threadSum);

  __shared__ float threadSums[BLOCK_DIM];
  threadSums[threadIdx.x] = threadSum;
  __syncthreads();

  // D) Offset each thread's elements by all preceding threads' sum.
  if (threadIdx.x > 0) {
    float prevPartialSum = threadSums[threadIdx.x - 1];
    for (unsigned c = 0; c < COARSE_FACTOR; ++c)
      buffer_s[threadSegment + c] += prevPartialSum;
  }
  __syncthreads();

  // E) Coalesced, strided store back to global memory.
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    unsigned long long g =
        (unsigned long long)blockSegment + c * BLOCK_DIM + threadIdx.x;
    if (g < N)
      output[g] = buffer_s[c * BLOCK_DIM + threadIdx.x];
  }

  // Record this tile's grand total (last thread holds the inclusive total).
  if (blockSums && threadIdx.x == BLOCK_DIM - 1)
    blockSums[blockIdx.x] = threadSums[BLOCK_DIM - 1];
}

// ====================================================================
// PASS 3 — add each block's exclusive prefix to every element it owns.
// blockSums has already been scanned (PASS 2), so blockSums[blockIdx-1]
// is the total of all elements in earlier blocks. Block 0 adds nothing.
// ====================================================================
__global__ void add_block_offsets(float *data, const float *scannedBlockSums,
                                  unsigned int N) {
  if (blockIdx.x == 0)
    return; // first block needs no offset
  float offset = scannedBlockSums[blockIdx.x - 1];
  unsigned int blockSegment = (unsigned long long)blockIdx.x * TILE;
  for (unsigned c = 0; c < COARSE_FACTOR; ++c) {
    unsigned long long g =
        (unsigned long long)blockSegment + c * BLOCK_DIM + threadIdx.x;
    if (g < N)
      data[g] += offset;
  }
}

// --------------------------------------------------------------------
// Host driver: hierarchically scan `data` of length n in place on device.
// Recurses on the block-sum array until it fits in a single tile.
// --------------------------------------------------------------------
static void deviceScanInPlace(float *d_data, unsigned long long n) {
  unsigned long long numBlocks = (n + TILE - 1) / TILE;

  if (numBlocks == 1) {
    // Single tile: one block fully scans it, no offsets needed.
    scan_kernel<<<1, BLOCK_DIM>>>(d_data, d_data, nullptr, (unsigned)n);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // PASS 1: scan every tile, collecting per-block totals.
  float *d_blockSums;
  CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(float)));
  scan_kernel<<<(unsigned)numBlocks, BLOCK_DIM>>>(d_data, d_data, d_blockSums,
                                                  (unsigned)n);
  CUDA_CHECK(cudaGetLastError());

  // PASS 2: scan the block totals (recurse — may itself span many tiles).
  deviceScanInPlace(d_blockSums, numBlocks);

  // PASS 3: add each block's exclusive prefix back into its tile.
  add_block_offsets<<<(unsigned)numBlocks, BLOCK_DIM>>>(d_data, d_blockSums,
                                                        (unsigned)n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_blockSums));
}

int main() {
  const unsigned long long N = 1000000000ULL; // 1 billion
  const size_t bytes = N * sizeof(float);

  printf("Scanning %llu elements (%.2f GB per buffer)\n", N, bytes / 1e9);

  // Host input. Values kept small (i % 7, range 0..6) so the running sum
  // stays exactly representable in float well past 1e9 elements is NOT
  // guaranteed — see the accuracy note below. We verify a prefix region.
  float *h_data = (float *)malloc(bytes);
  if (!h_data) {
    fprintf(stderr, "host malloc failed\n");
    return 1;
  }
  for (unsigned long long i = 0; i < N; ++i)
    h_data[i] = (float)(i % 7);

  float *d_data;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  deviceScanInPlace(d_data, N);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Copy back the result.
  float *h_out = (float *)malloc(bytes);
  CUDA_CHECK(cudaMemcpy(h_out, d_data, bytes, cudaMemcpyDeviceToHost));

  // Verify the first chunk exactly against a double-precision CPU scan.
  // (float accumulates rounding error over a billion adds, so we check a
  //  prefix where the GPU's float result still matches a float CPU scan.)
  const unsigned long long CHECK = 1u << 20; // first ~1M elements
  float accF = 0.0f;
  int mismatches = 0;
  for (unsigned long long i = 0; i < CHECK; ++i) {
    accF += h_data[i];
    if (h_out[i] != accF)
      ++mismatches;
  }

  // Report grand total in double for an order-of-magnitude sanity check.
  double accD = 0.0;
  for (unsigned long long i = 0; i < N; ++i)
    accD += h_data[i];

  double gelem = N / (ms / 1e3) / 1e9; // billions of elements/sec
  printf("Time            : %.2f ms  (%.2f G elem/s)\n", ms, gelem);
  printf("Prefix check    : first %llu elems, %d mismatches -> %s\n", CHECK,
         mismatches, mismatches ? "FAIL" : "PASS");
  printf("Grand total f32 : %.0f\n", h_out[N - 1]);
  printf("Grand total f64 : %.0f (reference)\n", accD);

  CUDA_CHECK(cudaFree(d_data));
  free(h_data);
  free(h_out);
  return 0;
}
