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
  };
  outputs = { nixpkgs, /*stable,*/ home-manager, catppuccin, zig, nix-index-database, ... }:
    let
      system = "x86_64-linux"; # replace with your system
      pkgs = import nixpkgs {inherit system;};
      zigpkgs = zig.packages."${system}";
      c = import ./config.nix;
      _secrets = builtins.tryEval (import ./secrets.nix);
      secrets = if _secrets.success then _secrets.value else {};
      # stablep = import stable {inherit system;config = {allowUnfree = true;};};
    in {
      homeConfigurations."misile" = home-manager.lib.homeManagerConfiguration {
        inherit pkgs;
        modules = [
          ./home.nix
          catppuccin.homeManagerModules.catppuccin
          nix-index-database.hmModules.nix-index
        ];
        extraSpecialArgs = {
          inherit c;
          inherit secrets;
          # inherit stablep;
          inherit zigpkgs;
        };
      };
    };
}
