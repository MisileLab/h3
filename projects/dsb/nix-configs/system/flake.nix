{
  description = "A simple NixOS flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    # stablep.url = "github:nixos/nixpkgs/nixos-unstable";
    # lanzaboote = {
    #   url = "github:nix-community/lanzaboote/v0.4.1";
    #   inputs.nixpkgs.follows = "nixpkgs";
    # };
  };

  outputs = { self, nixpkgs, /*stablep, *//*lanzaboote*/... }@_: {
    nixosConfigurations.nixos = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        # {_module.args = {stablep = import stablep { system = "x86_64-linux"; };};}
        ({ ... }: {
          nixpkgs.overlays = [(final: prev: {
            # FIX: https://github.com/NixOS/nixpkgs/issues/392278
            auto-cpufreq = prev.auto-cpufreq.overrideAttrs (oldAttrs: {
              postPatch =
                oldAttrs.postPatch
                + ''

                  substituteInPlace pyproject.toml \
                  --replace-fail 'psutil = "^6.0.0"' 'psutil = ">=6.0.0,<8.0.0"'
                '';
            });
          })];
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
