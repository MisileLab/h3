{config, pkgs, stablep, ...}:
let
  # electron-waylandify
  ewl = name: binaryPath: (pkgs.writeShellScriptBin "${name}" ''
    exec ${binaryPath} --enable-features=UseOzonePlatform --ozone-platform=wayland --enable-wayland-ime
  '');
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
      (ewl "figma" "${stablep.figma-linux}/bin/figma-linux")
      (ewl "discord" "${stablep.vesktop}/bin/vesktop")
      (ewl "vscode" "${stablep.vscodium}/bin/codium")
      (ewl "tetrio" "${stablep.tetrio-desktop.override{withTetrioPlus=true;}}/bin/tetrio")
      (ewl "bruno" "${stablep.bruno}/bin/bruno")
      (ewl "joplin" "${stablep.joplin-desktop}/bin/joplin-desktop")
      (ewl "signal" "${stablep.signal-desktop}/bin/signal-desktop")
      (ewl "element" "${stablep.element-desktop}/bin/element-desktop")
      (ewl "simplex" "${stablep.simplex-chat-desktop}/bin/simplex-chat-desktop")
      (ewl "slack" "${stablep.slack}/bin/slack")
      (ewl "chrome" "${stablep.ungoogled-chromium}/bin/chromium")
      stablep.ungoogled-chromium
    ];
  };
  programs = {
    joplin-desktop = {
      enable = true;
      package = stablep.joplin-desktop;
    };
    vscode = {
      enable = true;
      package = stablep.vscode-with-extensions.override { vscodeExtensions = with stablep.vscode-extensions; [ ms-vsliveshare.vsliveshare ]; };
    };
  };
}
