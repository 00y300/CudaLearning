# CUDA Kernel Progression — PMPP Chapters

| Chapter         | Topic            | Kernel                      | Current State | Goal                                   | New Concept                                       |
| --------------- | ---------------- | --------------------------- | ------------- | -------------------------------------- | ------------------------------------------------- |
| **Foundation**  | Thread hierarchy | Vector Add                  | ✅ Done       | Understand grid/block/thread indexing  | `blockIdx`, `threadIdx`, bounds check             |
| **Foundation+** | 2D indexing      | **Matrix Multiply (naive)** | ⬜            | Map 2D output to 2D thread grid        | Row/col indexing, dot product loop                |
| **Ch 7**        | Convolution      | Tiled Convolution           | ✅ Done       | Load input+halo into shared memory     | `__shared__`, `__syncthreads()`, halo cells       |
| **Ch 8**        | Stencil          | 3D Stencil                  | ✅ Done       | Extend tiling to 3 dimensions          | 3D grids, ghost cells, register caching           |
| **Ch 9**        | Histogram        | Privatized Histogram        | ✅ Done       | Eliminate atomic contention            | `atomicAdd`, per-block privatization, merge       |
| **Ch 10**       | Reduction        | Parallel Reduction          | ⬜            | Halve active threads each step         | Tree reduction, warp shuffle (`__shfl_down_sync`) |
| **Ch 11**       | Scan             | Prefix Sum                  | ⬜            | All-prefix-sums in O(n) work           | Kogge-Stone, Brent-Kung, work efficiency          |
| **Ch 12**       | Filter           | Stream Compaction           | ⬜            | Remove elements without branching      | Scan-based scatter, unknown output size           |
| **Ch 13**       | Merge            | Parallel Merge              | ⬜            | Merge two sorted arrays                | Co-rank function, load balancing across blocks    |
| **Ch 14**       | Sorting          | Radix Sort                  | ⬜            | Full GPU sort pipeline                 | Multi-pass digit sort, scan as primitive          |
| **Ch 15**       | MatMul Optimised | Tiled GEMM                  | ⬜            | Return to MatMul — now with everything | Register tiling, double buffering, tensor cores   |

## Why this order matters

- **Foundation → Ch 7**: shared memory pays off most when reuse is high — matmul and convolution both reuse inputs O(N) times
- **Ch 10 → Ch 11**: scan is built on reduction; do them back to back
- **Ch 11 → Ch 12/13**: filter and merge both use scan as a sub-primitive
- **Ch 15**: closing the loop on matmul shows how far you've come — naive vs tiled vs tensor-core is a story recruiters understand immediately
