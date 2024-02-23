{
  description = "Home Manager configuration of misile";

  inputs = {
    # Specify the source of Home Manager and Nixpkgs.
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    catppuccin.url = "github:Stonks3141/ctp-nix"; # will be changed to github:catppuccin/nix when avaliable
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  options = {
    desktop = lib.mkOption {
      type = lib.types.bool;
    };
  };

  config = {
    desktop = true;
  };

  outputs = { nixpkgs, home-manager, catppuccin, config, ... }:
    let
      system = "aarch64-linux"; # replace with your system
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [ ./home.nix catppuccin.homeManagerModules.catppuccin ] ++ (if config.desktop then [./desktop.nix] else []);
      };
    };
}
