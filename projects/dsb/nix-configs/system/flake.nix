{
  description = "A simple NixOS flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    lanzaboote = {
      url = "github:nix-community/lanzaboote/v0.4.1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, lanzaboote, ... }@inputs: {
    nixosConfigurations.nixos = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix

        lanzaboote.nixosModules.lanzaboote
        ({ pkgs, lib, ... }: {

          environment.systemPackages = [pkgs.sbctl];

          # Lanzaboote currently replaces the systemd-boot module.
          # This setting is usually set to true in configuration.nix
          # generated at installation time. So we force it to false
          # for now.
          boot = {
            loader.systemd-boot.enable = lib.mkForce false;
            lanzaboote = {
              enable = true;
              pkiBundle = "/etc/secureboot";
            };
          };
        })
      ];
    };
  };
}
