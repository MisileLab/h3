{config, pkgs, ...}:
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
      (ewl "figma" "${pkgs.figma-linux}/bin/figma-linux")
      (ewl "discord" "${pkgs.vesktop}/bin/vesktop")
      (ewl "vscode" "${pkgs.vscodium}/bin/codium")
      (ewl "tetrio" "${pkgs.tetrio-desktop.override{withTetrioPlus=true;}}/bin/tetrio")
      (ewl "insomnia" "${pkgs.bruno}/bin/bruno")
      (ewl "joplin" "${pkgs.joplin-desktop}/bin/joplin-desktop")
      (ewl "signal" "${pkgs.signal-desktop}/bin/signal-desktop")
      (ewl "element" "${pkgs.element-desktop}/bin/element-desktop")
      (ewl "simplex" "${pkgs.simplex-chat-desktop}/bin/simplex-chat-desktop")
      (ewl "slack" "${pkgs.slack}/bin/slack")
      (ewl "chromium" "${pkgs.ungoogled-chromium}/bin/chromium")
    ];
  };
  programs = {
    joplin-desktop.enable = true;
    vscode = {
      enable = true;
      package = pkgs.vscode-with-extensions.override { vscodeExtensions = with pkgs.vscode-extensions; [ ms-vsliveshare.vsliveshare ]; };
    };
  };
}
