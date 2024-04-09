{
  description = "Home Manager configuration of misile";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    catppuccin.url = "github:catppuccin/nix";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, home-manager, catppuccin, ... }:
    let
      system = "x86_64-linux"; # replace with your system
      pkgs = nixpkgs.legacyPackages.${system};
      c = import ./config.nix;
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [ ./home.nix ./desktop.nix catppuccin.homeManagerModules.catppuccin ];
      };
    };
}
