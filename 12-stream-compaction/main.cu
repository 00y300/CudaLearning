// stable_filter.cu
// Chapter 12 — Stable (order-preserving) stream compaction via scan-based scatter.
//
// Pipeline:  cond() -> keep flag  ->  exclusive scan of flags -> offset  ->  scatter.
// Stability comes from the scan, not from atomics: a kept key's output slot is a
// deterministic function of its input index (how many kept keys precede it), so the
// relative input order is preserved.
//
// The exclusive scan is a standard MULTI-LEVEL (recursive) grid scan:
//   1. blockExclusiveScan: each block exclusive-scans its own chunk and emits the
//      chunk's total into blockSums[].
//   2. Recurse: exclusive-scan blockSums[] the same way (any number of levels deep),
//      giving each block its base offset.
//   3. addBlockOffsets: add each block's base back into its elements.
//   4. scatter: kept threads write to output[offset]; last thread writes total size.
//
// This handles arbitrary N (no cap on the number of blocks).
//
// Build:  nvcc -O3 -arch=sm_120 stable_filter.cu -o stable_filter
// Run:    ./stable_filter [N]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_DIM 256

// ------------------------- error-checking helper -------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ------------------------- the filter condition --------------------------
// Keep even numbers. __host__ __device__ so the CPU reference and GPU kernel
// use the exact same predicate.
__host__ __device__ __forceinline__ bool cond(unsigned int v) {
    return (v % 2u) == 0u;
}

// =========================================================================
// Map keep flags: flags[i] = cond(input[i]) ? 1 : 0.  Separated out so the
// scan machinery below is a generic scan over an unsigned int array.
// =========================================================================
__global__ void computeFlagsKernel(const unsigned int* input,
                                   unsigned int* flags,
                                   unsigned int  N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) flags[i] = cond(input[i]) ? 1u : 0u;
}

// =========================================================================
// Generic per-block EXCLUSIVE scan (Kogge-Stone in shared memory).
// - in[]  : length n
// - out[] : exclusive prefix sums, per block (block-local, not yet global)
// - blockSums[] (optional): total of each block, one entry per block.
//   Pass nullptr when you don't need block totals (the top recursion level).
// =========================================================================
__global__ void blockExclusiveScan(const unsigned int* in,
                                    unsigned int* out,
                                    unsigned int* blockSums,
                                    unsigned int  n) {
    __shared__ unsigned int temp[BLOCK_DIM];

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + t;

    unsigned int v = (i < n) ? in[i] : 0u;
    temp[t] = v;
    __syncthreads();

    // Inclusive Kogge-Stone scan.
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int add = 0u;
        if (t >= stride) add = temp[t - stride];
        __syncthreads();
        temp[t] += add;
        __syncthreads();
    }

    unsigned int inclusive = temp[t];
    if (i < n) out[i] = inclusive - v;             // inclusive -> exclusive

    // Last thread emits this block's total (inclusive sum of the whole block).
    if (blockSums != nullptr && t == blockDim.x - 1) {
        blockSums[blockIdx.x] = inclusive;
    }
}

// =========================================================================
// Add each block's base offset (from the scanned blockSums) back into its
// elements, turning block-local exclusive scans into a global exclusive scan.
// =========================================================================
__global__ void addBlockOffsets(unsigned int* data,
                                 const unsigned int* blockBase,
                                 unsigned int  n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += blockBase[blockIdx.x];
}

// =========================================================================
// Recursive multi-level exclusive scan of an arbitrary-length array.
// After this returns, d_out[] holds the global exclusive prefix sum of d_in[].
// d_in is not modified. Caller owns d_in and d_out (length n each).
// =========================================================================
void deviceExclusiveScan(const unsigned int* d_in,
                         unsigned int* d_out,
                         unsigned int  n) {
    unsigned int numBlocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    if (numBlocks == 1) {
        // Base case: one block scans the whole array, no block sums needed.
        blockExclusiveScan<<<1, BLOCK_DIM>>>(d_in, d_out, nullptr, n);
        return;
    }

    // Recursive case: scan each block, collect block totals, scan those,
    // then fold the scanned totals back in.
    unsigned int *d_blockSums, *d_blockBase;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_blockBase, numBlocks * sizeof(unsigned int)));

    blockExclusiveScan<<<numBlocks, BLOCK_DIM>>>(d_in, d_out, d_blockSums, n);

    // Exclusive-scan the block totals (this is where the recursion happens).
    deviceExclusiveScan(d_blockSums, d_blockBase, numBlocks);

    // Add each block's base offset back into its elements.
    addBlockOffsets<<<numBlocks, BLOCK_DIM>>>(d_out, d_blockBase, n);

    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_blockBase));
}

// =========================================================================
// Scatter: kept threads write their value to output[offset]. The last input
// thread records the total output size = its exclusive offset + its own flag.
// =========================================================================
__global__ void scatterKernel(const unsigned int* input,
                               const unsigned int* offsets,
                               unsigned int* output,
                               unsigned int* d_outputSize,
                               unsigned int  N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int val    = input[i];
    unsigned int keep   = cond(val) ? 1u : 0u;
    unsigned int offset = offsets[i];

    if (keep) output[offset] = val;

    if (i == N - 1) *d_outputSize = offset + keep;
}

// ------------------------- CPU reference ---------------------------------
unsigned int cpuStableFilter(const std::vector<unsigned int>& in,
                             std::vector<unsigned int>& out) {
    unsigned int count = 0;
    for (unsigned int v : in) if (cond(v)) out[count++] = v;
    return count;
}

// =========================================================================
int main(int argc, char** argv) {
    unsigned int N = (argc > 1) ? (unsigned int)strtoul(argv[1], nullptr, 10)
                                : (1u << 20);   // default ~1M elements

    unsigned int numBlocks = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    printf("N = %u, BLOCK_DIM = %u, numBlocks = %u\n", N, BLOCK_DIM, numBlocks);

    // ---- host input ----
    std::vector<unsigned int> h_input(N);
    srand(1234);
    for (unsigned int i = 0; i < N; ++i) h_input[i] = rand() % 1000;

    // ---- CPU reference ----
    std::vector<unsigned int> h_ref(N);
    unsigned int refCount = cpuStableFilter(h_input, h_ref);

    // ---- device allocations ----
    unsigned int *d_input, *d_output, *d_flags, *d_offsets, *d_outputSize;
    CUDA_CHECK(cudaMalloc(&d_input,      N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_output,     N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_flags,      N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_offsets,    N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_outputSize, sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // ---- timing setup ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto runPipeline = [&]() {
        computeFlagsKernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_flags, N);
        deviceExclusiveScan(d_flags, d_offsets, N);
        scatterKernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_offsets,
                                                d_output, d_outputSize, N);
    };

    // Warm-up (first launch pays one-time init costs).
    runPipeline();
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- timed run ----
    CUDA_CHECK(cudaEventRecord(start));
    runPipeline();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // ---- copy results back ----
    unsigned int gpuCount = 0;
    CUDA_CHECK(cudaMemcpy(&gpuCount, d_outputSize,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> h_output(N);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // ---- correctness check (count AND order) ----
    bool ok = (gpuCount == refCount);
    if (ok) {
        for (unsigned int i = 0; i < refCount; ++i) {
            if (h_output[i] != h_ref[i]) { ok = false; break; }
        }
    }

    printf("CPU kept: %u\n", refCount);
    printf("GPU kept: %u\n", gpuCount);
    printf("Result:   %s\n", ok ? "PASS (order preserved)" : "FAIL");
    printf("GPU time: %.3f ms  (%.2f Melem/s)\n", ms, N / (ms * 1e3));

    // ---- cleanup ----
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_flags);
    cudaFree(d_offsets); cudaFree(d_outputSize);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
