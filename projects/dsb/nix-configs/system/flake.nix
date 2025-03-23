{
  description = "A simple NixOS flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    stablep.url = "github:nixos/nixpkgs/nixos-unstable";
    # lanzaboote = {
    #   url = "github:nix-community/lanzaboote/v0.4.1";
    #   inputs.nixpkgs.follows = "nixpkgs";
    # };
  };

  outputs = { self, nixpkgs, stablep, /*lanzaboote*/... }@_: {
    nixosConfigurations.nixos = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        # https://github.com/NixOS/nixpkgs/issues/392278
        ({ stablep, ... }: {
          nixpkgs.overlays = [ (final: prev: {final.auto-cpufreq = stablep.auto-cpufreq;}) ];
        })
        # lanzaboote.nixosModules.lanzaboote
        ({ pkgs, lib, ... }: {
          environment.systemPackages = [pkgs.sbctl];
          # boot = {
          #   loader.systemd-boot.enable = lib.mkForce false;
          #   lanzaboote = {
          #     enable = true;
          #     pkiBundle = "/etc/secureboot";
          #   };
          # };
        })
      ];
    };
  };
}
