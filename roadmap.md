# CUDA Kernel Progression — PMPP Chapters

| Chapter         | Topic             | Kernel                      | Current State | Goal                                       | New Concept                                                               |
| --------------- | ----------------- | --------------------------- | ------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| **Foundation**  | Thread hierarchy  | Vector Add                  | ✅ Done       | Understand grid/block/thread indexing      | `blockIdx`, `threadIdx`, bounds check                                     |
| **Foundation+** | 2D indexing       | **Matrix Multiply (naive)** | ⬜            | Map 2D output to 2D thread grid            | Row/col indexing, dot product loop                                        |
| **Ch 7**        | Convolution       | Tiled Convolution           | ✅ Done       | Load input+halo into shared memory         | `__shared__`, `__syncthreads()`, halo cells                               |
| **Ch 8**        | Stencil           | 3D Stencil                  | ✅ Done       | Extend tiling to 3 dimensions              | 3D grids, ghost cells, register caching                                   |
| **Ch 9**        | Histogram         | Privatized Histogram        | ✅ Done       | Eliminate atomic contention                | `atomicAdd`, per-block privatization, merge                               |
| **Ch 10**       | Reduction         | Parallel Reduction (×4)     | ✅ Done       | Full tree down-sweep + warp-shuffle hybrid | Tree reduction, `__shfl_down_sync`, up-sweep vs down-sweep strides        |
| **Ch 11**       | Scan              | Prefix Sum                  | ✅ Done       | Hierarchical 3-pass global scan            | Coarse loading, warp-level scan, block-level scan, block offset stitching |
| **Ch 12**       | Stream Compaction | Stream Compaction           | ⬜            | Remove elements without branching          | Scan-based scatter, unknown output size                                   |
| **Ch 13**       | Merge             | Parallel Merge              | ⬜            | Merge two sorted arrays                    | Co-rank function, load balancing across blocks                            |
| **Ch 14**       | Sorting           | Radix Sort                  | ⬜            | Full GPU sort pipeline                     | Multi-pass digit sort, scan as primitive                                  |
| **Ch 15**       | MatMul Optimised  | Tiled GEMM                  | ⬜            | Return to MatMul — now with everything     | Register tiling, double buffering, tensor cores                           |

## Why This Order Matters

- **Foundation → Ch 7**: shared memory pays off most when reuse is high — matmul and convolution both reuse inputs O(N) times
- **Ch 10 → Ch 11**: scan is built on reduction; do them back to back
- **Ch 11 → Ch 12/13**: filter and merge both use scan as a sub-primitive
- **Ch 15**: closing the loop on matmul shows how far you've come — naive vs tiled vs tensor-core is a story recruiters understand immediately
