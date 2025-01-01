{
  description = "A Nix-flake-based C/C++ development environment";

  inputs.nixpkgs.url = "github:nixos/nixpkgs";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell.override
          {
            stdenv = pkgs.clangStdenv;
          }
          {
            packages = with pkgs; [
              clang-tools
              fish
              php
            ];
            shellHook = ''
              if (UV_ACTIVATED != 1) {
                env UV_ACTIVATED=1 ${pkgs.uv}/bin/uv run ${pkgs.nushell}/bin/nu
                exit
              }
            '';
          };
      });
    };
}
