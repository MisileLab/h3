{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # Common build inputs for Tauri
        buildInputs = with pkgs; [
          # Tauri dependencies
          webkitgtk_4_1
          gtk3
          cairo
          gdk-pixbuf
          glib
          dbus
          openssl
          librsvg
          libsoup_3

          # Additional dependencies
          at-spi2-atk
          atkmm
          gobject-introspection
          harfbuzz
          pango
        ];

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          nodejs
          pnpm
          cargo-tauri
        ];

        # Library path for runtime
        libraryPath = pkgs.lib.makeLibraryPath buildInputs;
      in
      {
        devShells.default = pkgs.mkShell {
          inherit buildInputs nativeBuildInputs;

          shellHook = ''
            export LD_LIBRARY_PATH="${libraryPath}:$LD_LIBRARY_PATH"
            export GIO_MODULE_DIR="${pkgs.glib-networking}/lib/gio/modules/"
            export WEBKIT_DISABLE_COMPOSITING_MODE=1
          '';
        };
      }
    );
}
