{config, pkgs, ...}:
let
  # electron-waylandify
  ewl = binaryPath: ''
    exec ${binaryPath} --enable-features=UseOzonePlatform --ozone-platform=wayland --enable-wayland-ime
  '';
in
{
  home = {
    file = {
      ".config/joplin-desktop/userstyle.css".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/joplin";
        rev="b0a886ce7ba71b48fdbf72ad27f3446400ebcdb9";
      }}/src/mocha/userstyle.css";
    };
    packages = [
      (pkgs.writeShellScriptBin "figma" (ewl "${pkgs.figma-linux}/bin/figma-linux"))
      (pkgs.writeShellScriptBin "discord" (ewl "${pkgs.vesktop}/bin/vesktop"))
      (pkgs.writeShellScriptBin "vscode" (ewl "${pkgs.vscodium}/bin/codium"))
      (pkgs.writeShellScriptBin "gdb" "${pkgs.pwndbg}/bin/pwndbg")
      (pkgs.writeShellScriptBin "tetrio" (ewl "${pkgs.tetrio-desktop.override{withTetrioPlus=true;}}/bin/tetrio"))
      (pkgs.writeShellScriptBin "insomnia" (ewl "${pkgs.bruno}/bin/bruno"))
      (pkgs.writeShellScriptBin "joplin" (ewl "${pkgs.joplin-desktop}/bin/joplin-desktop"))
      (pkgs.writeShellScriptBin "signal" (ewl "${pkgs.signal-desktop}/bin/signal-desktop"))
    ];
  };
  programs = {
    joplin-desktop.enable=true;
    vscode = {
      enable = true;
      package = pkgs.vscodium;
    };
  };
}
