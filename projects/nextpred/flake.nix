{
  description = "Next Action Predictor - Cross-platform OS-level action pattern learning and suggestion system";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.dbus ];
          buildInputs = with pkgs; [
            dbus
            pkg-config
            glib
          ];
          nativeBuildInputs = with pkgs; [
            dbus.dev
            pkg-config
            glib.dev
          ];
      };
    });
}
