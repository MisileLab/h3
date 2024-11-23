{
  description = "An empty flake template that you can adapt to your own environment";

  # Flake inputs
  inputs.nixpkgs.url = "github:nixos/nixpkgs";

  # Flake outputs
  outputs = { self, nixpkgs }:
    let
      # The systems supported for this flake
      supportedSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
        "aarch64-linux" # 64-bit ARM Linux
        "x86_64-darwin" # 64-bit Intel macOS
        "aarch64-darwin" # 64-bit ARM macOS
      ];

      # Helper to provide system-specific attributes
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          # The Nix packages provided in the environment
          # Add any you need here
          packages = with pkgs; [ python312Full ];

          # Set any environment variables for your dev shell
          env = {
          };

          # Add any shell logic you want executed any time the environment is activated
          shellHook = ''
            if test -n "$FISH_VERSION"; then
              source ./.venv/bin/activate.fish
            elif test -n "$CSH_VERSION"; then
              source ./.venv/bin/activate.csh
            elif test -n "$NU_VERSION"; then
              source ./.venv/bin/activate.nu
            else
              source ./.venv/bin/activate
            fi
          '';
        };
      });
    };
}
