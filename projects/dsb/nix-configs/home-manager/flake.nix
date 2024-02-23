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

  outputs = { nixpkgs, home-manager, catppuccin, ... }:
    let
      system = "aarch64-linux"; # replace with your system
      pkgs = nixpkgs.legacyPackages.${system};
      c = {
        desktop = true;
      };
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [ ./home.nix {inherit c;} catppuccin.homeManagerModules.catppuccin ] ++ (if c.desktop then [./desktop.nix] else []);
      };
    };
}
