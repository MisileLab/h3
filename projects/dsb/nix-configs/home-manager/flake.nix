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
    #sops-nix.url = "github:Mic92/sops-nix";
  };

  outputs = { nixpkgs, home-manager, catppuccin, ... }:
    let
      system = "x86_64-linux"; # replace with your system
      pkgs = nixpkgs.legacyPackages.${system};
      c = import ./config.nix;
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [ ./home.nix catppuccin.homeManagerModules.catppuccin /*sops-nix.homeManagerModules.sops*/ ] ++ (if c.desktop then [./desktop.nix] else []);
      };
    };
}
