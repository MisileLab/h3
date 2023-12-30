{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        overlays = [
          rust-overlay.overlays.default
          (final: prev: {
            rustToolchain = prev.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          })
        ];
        supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
        forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
          pkgs = import nixpkgs { inherit overlays system; };
        });

        libraries = with pkgs; [
          webkitgtk
          gtk3
          cairo
          gdk-pixbuf
          glib.out
          dbus.lib
          openssl.out
        ];
      in
      {
        devShell = pkgs.mkShell {
          packages = with pkgs; [
          pkg-config
          dbus
          openssl
          glib
          gtk3
          libsoup
          webkitgtk
          appimagekit
          nodePackages_latest.pnpm
          rustToolchain
          ];

          shellHook =
            let
              joinLibs = libs: builtins.concatStringsSep ":" (builtins.map (x: "${x}/lib") libs);
              libs = joinLibs libraries;
            in
            ''
              export LD_LIBRARY_PATH=${libs}:$LD_LIBRARY_PATH
            '';
        };
      });
}
