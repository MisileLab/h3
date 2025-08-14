{
  description = "Home Manager configuration of misile";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    zig = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # stable.url = "github:nixos/nixpkgs/nixos-unstable";
    # nur.url = "github:nix-community/NUR";
    catppuccin = {
      url = "github:catppuccin/nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nix-index-database = {
      url = "github:nix-community/nix-index-database";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    sops-nix = {
      url = "github:Mic92/sops-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    niri = {
      url = "github:sodiboo/niri-flake";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { nixpkgs, /*stable, */home-manager, catppuccin, zig, nix-index-database, sops-nix, niri, ... }:
    let
      system = "x86_64-linux"; # replace with your system
      pkgs = import nixpkgs {inherit system;};
      zigpkgs = zig.packages."${system}";
      c = import ./config.nix;
      # stablep = import stable {inherit system;config = {allowUnfree = true;};};
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [
          ./home.nix
          catppuccin.homeModules.catppuccin
          nix-index-database.homeModules.nix-index
          sops-nix.homeManagerModules.sops
          niri.homeModules.niri
        ];
        extraSpecialArgs = {
          inherit c;
          # inherit stablep;
          inherit zigpkgs;
        };
      };
    };
}
