{
  description = "Home Manager configuration of misile";

  inputs = {
    # https://github.com/NixOS/nixpkgs/pull/357119
    nixpkgs.url = "github:misilelab/nixpkgs/lunarvim";
    stable.url = "github:nixos/nixpkgs/nixos-unstable";
    catppuccin.url = "github:catppuccin/nix";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    c = {
      url = "path:./config.nix";
      flake = false;
    };
  };
  outputs = { nixpkgs, stable, home-manager, catppuccin, c, ... }:
    let
      system = "x86_64-linux"; # replace with your system
      pkgs = import nixpkgs {inherit system;};
      c = import ./config.nix;
      stablep = import stable {inherit system;config = {allowUnfree = true;};};
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [
          ./home.nix
          catppuccin.homeManagerModules.catppuccin
        ];
        extraSpecialArgs = {
          inherit c;
          inherit stablep;
        };
      };
    };
}
