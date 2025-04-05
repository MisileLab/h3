{
  description = "A simple NixOS flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
#    stable.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, /*stable,*/ ... }@inputs: {
    nixosConfigurations.veritas = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules =
      let
        defaults = { pkgs, ... }: {
          #_module.args.stable = import inputs.stable { inherit (pkgs.stdenv.targetPlatform) system; };
        };
        #attic = import attic;
      in
      [
        defaults
        ./configuration.nix
      ];
    };
  };
}
