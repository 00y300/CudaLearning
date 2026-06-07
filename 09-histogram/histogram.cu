// ch9_histogram.cu — Privatized histogram of an image
//
// Build: cmake -B build -S . && cmake --build build --target 09-histogram
// Run:   ./build/09-histogram
//
// Loads an image, builds a per-channel (R,G,B) histogram using the
// privatization technique from PMPP Ch.9 to eliminate atomicAdd
// contention, then compares against a naive (fully-atomic) baseline.

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 256

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

// ─── Naive kernel: one atomicAdd per pixel ───────────────────────────
__global__ void hist_naive(const unsigned char* img, unsigned int* hist, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        atomicAdd(&hist[img[i]], 1u);
}

// ─── Privatized kernel: per-block shared counters, then merge ────────
// Real privatization: a fixed-size grid where each thread strides over
// the whole image, accumulating into a per-block shared histogram.
// This minimizes global atomic contention (one atomicAdd per bin per
// block on merge instead of one per pixel).
__global__ void hist_privatized(const unsigned char* img, unsigned int* hist, int N) {
    __shared__ unsigned int block_hist[NUM_BINS];

    // Initialize private histogram to zero
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        block_hist[b] = 0u;
    __syncthreads();

    // Grid-stride loop: every thread covers many pixels regardless of grid size
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        atomicAdd(&block_hist[img[i]], 1u);
    }
    __syncthreads();

    // Merge this block's private histogram into the global histogram
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        atomicAdd(&hist[b], block_hist[b]);
}

// Helper: run naive + privatized for one channel, fill host result arrays
static void run_channel(const char* name,
                        const unsigned char* h_chan, int N,
                        unsigned int* out_naive, unsigned int* out_priv,
                        float* ms_naive_out, float* ms_priv_out) {
    unsigned char* d_chan;
    unsigned int *d_hist_naive, *d_hist_priv;
    CUDA_CHECK(cudaMalloc(&d_chan, N));
    CUDA_CHECK(cudaMalloc(&d_hist_naive, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_hist_priv,  NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_chan, h_chan, N, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    // Naive
    dim3 block_naive(256);
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x);
    CUDA_CHECK(cudaMemset(d_hist_naive, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(t0));
    hist_naive<<<grid_naive, block_naive>>>(d_chan, d_hist_naive, N);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out_naive, d_hist_naive, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(ms_naive_out, t0, t1);

    // Privatized: fixed grid sized to the device, grid-stride loop covers all pixels
    int dev = 0; cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    dim3 block_priv(256);
    dim3 grid_priv(prop.multiProcessorCount * 32);  // enough blocks to saturate the GPU
    CUDA_CHECK(cudaMemset(d_hist_priv, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(t0));
    hist_privatized<<<grid_priv, block_priv>>>(d_chan, d_hist_priv, N);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out_priv, d_hist_priv, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaEventElapsedTime(ms_priv_out, t0, t1);

    printf("  [%s] naive: %.3f ms  |  privatized: %.3f ms\n", name, *ms_naive_out, *ms_priv_out);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_chan); cudaFree(d_hist_naive); cudaFree(d_hist_priv);
}

int main() {
    const char* in_path = "assets/f1Car.jpg";

    int W, H, ch;
    unsigned char* img = stbi_load(in_path, &W, &H, &ch, 3);
    int img_is_malloc = 0;
    if (!img) {
        fprintf(stderr, "Failed to load %s — generating synthetic test image\n", in_path);
        W = 512; H = 512; ch = 3;
        img = (unsigned char*)malloc(W * H * 3);
        img_is_malloc = 1;
        for (int j = 0; j < W * H; j++) {
            img[j*3+0] = (unsigned char)(j % 256);          // R
            img[j*3+1] = (unsigned char)(j * 3 % 256);      // G
            img[j*3+2] = (unsigned char)(j * 7 % 256);      // B
        }
    }
    printf("Image: %s (%dx%d, %d channels)\n", in_path, W, H, ch);

    int N = W * H;  // pixels per channel
    printf("Pixels per channel for histogram: %d\n", N);

    // ─── Deinterleave channels on the host ───────────────────────────
    unsigned char *h_r = (unsigned char*)malloc(N);
    unsigned char *h_g = (unsigned char*)malloc(N);
    unsigned char *h_b = (unsigned char*)malloc(N);
    for (int i = 0; i < N; i++) {
        h_r[i] = img[i*3+0];
        h_g[i] = img[i*3+1];
        h_b[i] = img[i*3+2];
    }

    // Host result buffers
    unsigned int *naive_r = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    unsigned int *naive_g = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    unsigned int *naive_b = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    unsigned int *priv_r  = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    unsigned int *priv_g  = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));
    unsigned int *priv_b  = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));

    float ms_nr, ms_pr, ms_ng, ms_pg, ms_nb, ms_pb;
    printf("\nComputing histograms (all channels run on the GPU):\n");
    run_channel("R", h_r, N, naive_r, priv_r, &ms_nr, &ms_pr);
    run_channel("G", h_g, N, naive_g, priv_g, &ms_ng, &ms_pg);
    run_channel("B", h_b, N, naive_b, priv_b, &ms_nb, &ms_pb);

    // ─── Verify correctness ──────────────────────────────────────────
    int match_r = 1, match_g = 1, match_b = 1;
    for (int i = 0; i < NUM_BINS; i++) {
        if (naive_r[i] != priv_r[i]) match_r = 0;
        if (naive_g[i] != priv_g[i]) match_g = 0;
        if (naive_b[i] != priv_b[i]) match_b = 0;
    }
    printf("\nCorrectness (naive vs privatized): R=%s  G=%s  B=%s\n",
           match_r ? "PASS" : "FAIL",
           match_g ? "PASS" : "FAIL",
           match_b ? "PASS" : "FAIL");

    // Sum of all bins should equal pixel count for every channel
    unsigned long long sr = 0, sg = 0, sb = 0;
    for (int i = 0; i < NUM_BINS; i++) { sr += naive_r[i]; sg += naive_g[i]; sb += naive_b[i]; }
    printf("Bin sums  R=%llu  G=%llu  B=%llu  |  Expected each: %d\n", sr, sg, sb, N);

    // ─── Save histogram as a visual PNG ──────────────────────────────
    // Three stacked 256-wide bands (R on top, then G, then B), each
    // hist_h tall. Each channel is normalized to its own max so the
    // shape is visible regardless of absolute counts.
    int hist_w = NUM_BINS, hist_h = 100;
    int bands = 3;

    unsigned int* chans[3] = { naive_r, naive_g, naive_b };
    unsigned int  maxv[3]  = { 1u, 1u, 1u };
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < NUM_BINS; i++)
            if (chans[c][i] > maxv[c]) maxv[c] = chans[c][i];

    int total_h = hist_h * bands;
    unsigned char* hist_img = (unsigned char*)malloc((size_t)hist_w * total_h * 3);

    for (int y = 0; y < total_h; y++) {
        int band = y / hist_h;          // 0=R, 1=G, 2=B
        int yb   = y % hist_h;          // row within the band (0 at top)
        for (int x = 0; x < hist_w; x++) {
            int bar_h = (int)(((unsigned long long)chans[band][x] * hist_h) / maxv[band]);
            int filled = yb >= (hist_h - bar_h);
            int p = (y * hist_w + x) * 3;
            // stb writes RGB. Color each band by its own channel.
            unsigned char r = 0, g = 0, b = 0;
            if (filled) {
                if (band == 0) r = 255;
                else if (band == 1) g = 255;
                else b = 255;
            }
            hist_img[p+0] = r;
            hist_img[p+1] = g;
            hist_img[p+2] = b;
        }
    }

    char out_path[256];
    snprintf(out_path, sizeof(out_path), "assets/histogram_%dx%d.png", W, H);
    stbi_write_png(out_path, hist_w, total_h, 3, hist_img, hist_w * 3);
    printf("Saved histogram visualization: %s (%dx%d)\n", out_path, hist_w, total_h);

    // Cleanup
    free(h_r); free(h_g); free(h_b);
    free(naive_r); free(naive_g); free(naive_b);
    free(priv_r);  free(priv_g);  free(priv_b);
    free(hist_img);
    if (img_is_malloc) free(img); else stbi_image_free(img);

    printf("Done. Hardware: RTX 5090 (Blackwell, sm_120)\n");
    return 0;
}
