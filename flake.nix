{
  description = "CUDA C++ Development Environment with Clang Tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    nix-vscode-extensions.url = "github:nix-community/nix-vscode-extensions";
  };

  outputs =
    {
      self,
      nixpkgs,
      nix-vscode-extensions,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      cudaPackages = pkgs.cudaPackages_13_0;

      cudaPkgs = with cudaPackages; [
        cuda_nvcc
        cuda_cudart
        cccl
        cuda_gdb
        cuda_cuobjdump
        cuda_nvdisasm
        cuda_sanitizer_api
        cuda_nvtx
      ];

      cudaMerged = pkgs.symlinkJoin {
        name = "cuda-${cudaPackages.cudaMajorMinorVersion}-merged";
        paths = cudaPkgs;
      };

      llvm = pkgs.llvmPackages_20;

      # cpptools extension package — we pull OpenDebugAD7 directly from it
      # rather than walking back from the `code` wrapper at runtime.
      cpptools = pkgs.vscode-extensions.ms-vscode.cpptools;

      marketplace = nix-vscode-extensions.extensions.${system}.vscode-marketplace;
      stdExt =
        (with pkgs.vscode-extensions; [
          ms-vscode.cpptools
          vadimcn.vscode-lldb
        ])
        ++ [
          marketplace.nvidia.nsight-vscode-edition
          marketplace.vscodevim.vim
          marketplace.ms-vscode.cmake-tools
        ];
      vscodePkgs = pkgs.vscode-with-extensions.override {
        vscodeExtensions = stdExt;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "cuda-clang";
        packages =
          with pkgs;
          [
            vscodePkgs
            cmake
            gnumake
            gdb
            lldb
            llvm.clang-tools
            llvm.libstdcxxClang
          ]
          ++ cudaPkgs;

        shellHook = ''
          # Driver libs first (real libcuda.so.1 wins over toolkit stub)
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"

          # Unified CUDA root
          export CUDA_PATH="${cudaMerged}"
          export CUDA_HOME="$CUDA_PATH"
          export CUDAToolkit_ROOT="$CUDA_PATH"

          # cuda-gdb needs software preemption on single-GPU workstations
          export CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1

          # Expose OpenDebugAD7 directly so nvim-dap doesn't have to hunt for it.
          # Path within the cpptools extension is stable: extension/debugAdapters/bin/OpenDebugAD7
          export OPENDEBUGAD7_PATH="${cpptools}/share/vscode/extensions/ms-vscode.cpptools/debugAdapters/bin/OpenDebugAD7"

          # Sanity check at shell entry — fail loud if the layout changed
          if [ ! -x "$OPENDEBUGAD7_PATH" ]; then
            echo "warning: OpenDebugAD7 not at expected path: $OPENDEBUGAD7_PATH" >&2
            # try to locate it within the cpptools store path
            found=$(find "${cpptools}" -name OpenDebugAD7 -type f -executable 2>/dev/null | head -1)
            if [ -n "$found" ]; then
              export OPENDEBUGAD7_PATH="$found"
              echo "  -> found at: $OPENDEBUGAD7_PATH" >&2
            fi
          fi
        '';
      };
    };
}
