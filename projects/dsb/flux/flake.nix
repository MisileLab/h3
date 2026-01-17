{
  description = "Flux development environment with LLVM 21";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustToolchain = pkgs.rust-bin.stable.latest.default;
        llvmPkgs = pkgs.llvmPackages_21;
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustToolchain
            pkg-config
            cmake
            ninja
            rust-analyzer
            llvmPkgs.llvm
            llvmPkgs.clang
            llvmPkgs.lld
            libffi
            libxml2
          ];

          LLVM_SYS_211_PREFIX = "${llvmPkgs.llvm.dev}";
          LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
          LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [ pkgs.libffi pkgs.libxml2 ]}";
          CARGO_TARGET_DIR = "target";
        };
      });
}
