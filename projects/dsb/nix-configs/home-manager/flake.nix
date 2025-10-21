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
    nix-ai-tools = {
      url = "github:numtide/nix-ai-tools";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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
  outputs = { nixpkgs, /*stable, */home-manager, catppuccin, zig, nix-index-database, sops-nix, niri, nix-ai-tools, ... }:
    let
      system = "x86_64-linux"; # replace with your system
      overlays = [
        (final: prev: {
          opencode = nix-ai-toolspkgs.opencode;
          dart = prev.dart.overrideAttrs (old: {
            installPhase = ''
              runHook preInstall
              [ -f LICENSE ] && rm LICENSE
              cp -R . $out
            ''
            + final.lib.optionalString (final.stdenv.hostPlatform.isLinux) ''
              find $out/bin -executable -type f -exec patchelf --set-interpreter ${final.bintools.dynamicLinker} {} \;
            ''
            + ''
              runHook postInstall
            '';
          });
        })
      ];
      pkgs = nixpkgs.packages."${system}";
      nix-ai-toolspkgs = nix-ai-tools.packages."${system}";
      zigpkgs = zig.packages."${system}";
      c = import ./config.nix;
      # stablep = stable.packages."${system}";
    in {
      nixpkgs.overlays = overlays;
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
          inherit zigpkgs;
          # inherit stablep;
        };
      };
    };
}
