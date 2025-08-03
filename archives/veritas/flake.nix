{
  description = "slunchv2's infrastructure";

  inputs = {
    pkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    disko = {
      url = "github:nix-community/disko/latest";
      inputs.nixpkgs.follows = "pkgs";
    };
    nix-index-database = {
      url = "github:nix-community/nix-index-database";
      inputs.nixpkgs.follows = "pkgs";
    };
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "pkgs";
    };
    sops-nix = {
      url = "github:Mic92/sops-nix";
      inputs.nixpkgs.follows = "pkgs";
    };
  };
  outputs = { pkgs, home-manager, disko, sops-nix, ... }:
    {
      nixosConfigurations.veritas = pkgs.lib.nixosSystem {
        system = "x86_64-linux";
        modules = [
          ({lib, ...}: {
            nixpkgs = {
              overlays = [ (import ./pkgs) ];
              config.allowUnfreePredicate = pkg: builtins.elem (lib.getName pkg) [
                "steam-unwrapped"
              ];
            };
          })
          sops-nix.nixosModules.sops
          disko.nixosModules.disko
          home-manager.nixosModules.home-manager
          {
            home-manager = {
              useGlobalPkgs = false;
              useUserPackages = false;
              users.veritas = import ./home.nix;
            };
          }
          ./configuration.nix
          ./modules
        ];
      };
    };
}
