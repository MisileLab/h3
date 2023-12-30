{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay)
        (final: prev: {
          rustToolchain = prev.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        })];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

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
          buildInputs = with pkgs; [
            pkg-config
            dbus
            openssl
            glib
            gtk3
            libsoup
            webkitgtk
            appimagekit
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
