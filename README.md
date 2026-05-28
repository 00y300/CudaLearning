# CudaLearning

Collection of CUDA learning projects covering GPU kernel optimization patterns.
This repo was created to reinforce my understanding of [Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1) (Hwu, Kirk, El Hajj).

## Chapters

| Chapter | Description                          |
| ------- | ------------------------------------ |
| 07      | Tiled 2D convolution (Gaussian blur) |
| 08      | 3D stencil (7-point Laplacian)       |
| 09      | Parallel histogram                   |
| 10      | Parallel reduction                   |
| 11      | Prefix sum (scan)                    |
| 12      | Stream compaction                    |
| 13      | Parallel merge                       |
| 14      | Radix sort                           |
| 15      | Tiled GEMM                           |

## Development Environment

This project uses [Nix flakes](https://nixos.wiki/wiki/Flakes) to manage the development environment. Activate it with:

```bash
nix develop -c zsh
```

This provides `nvcc`, `cmake`, `clangd`, and all other build dependencies automatically.

## Build

```bash
cmake -B build -S .
cmake --build build
```

## Requirements

- NVIDIA GPU (compute capability 12.0+) — tested on RTX 5090 (Blackwell, sm_120)
- CUDA Toolkit 13.0
- Nix with flakes enabled (or CUDA Toolkit 13.0 + CMake 3.18+ for manual setup)
