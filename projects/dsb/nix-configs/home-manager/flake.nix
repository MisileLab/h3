{
  description = "Home Manager configuration of misile";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    # unstablepkgs.url = "github:nixos/nixpkgs/nixos-unstable";
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

  outputs = { nixpkgs, /*unstablepkgs,*/ home-manager, catppuccin, c, ... }:
    let
      system = "x86_64-linux"; # replace with your system
      pkgs = nixpkgs.legacyPackages.${system};
      c = import ./config.nix;
      # unstable-pkgs = unstablepkgs.legacyPackages.${system};
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [
          ./home.nix
          catppuccin.homeManagerModules.catppuccin
        ];
        extraSpecialArgs = {
          inherit c;
        #   inherit unstable-pkgs;
        };
      };
    };
}
