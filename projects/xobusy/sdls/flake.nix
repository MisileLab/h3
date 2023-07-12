{
  description = "SDL Nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, ... }@inputs: inputs.utils.lib.eachSystem [
    "x86_64-linux"
  ] (system: let
    pkgs = import nixpkgs {
      inherit system;
      overlays = [];
    };
  in {
    devShells.default = pkgs.mkShell rec {
      name = "Nix SDL";

      packages = with pkgs; [
        llvmPackages_16.clang
        SDL2
      ];

      shellHook = let
        icon = "f121";
      in ''
        export PS1="$(echo -e '\u${icon}') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (${name}) \\$ \[$(tput sgr0)\]"
        ./start.sh
      '';
    };

    packages.default = pkgs.callPackage ./default.nix {};
  });
}