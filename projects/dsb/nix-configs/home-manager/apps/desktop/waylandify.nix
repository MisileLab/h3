{config, pkgs, ...}:
let
  # base
  base = name: binaryPath: args: (pkgs.writeShellScriptBin "${name}" ''
    exec ${binaryPath} ${args}
  '');
  # electron-waylandify
  ewl = name: binaryPath: base name binaryPath "--enable-features=UseOzonePlatform,WaylandWindowDecorations --ozone-platform-hint=auto --ozone-platform=wayland --enable-wayland-ime";
  # java-waylandify
  jwl = name: binaryPath: base name binaryPath "-Dawt.toolkit.name=WLToolkit";
in
{
  home = {
    file = {
      ".config/joplin-desktop/userstyle.css".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/joplin";
        rev="b0a886ce7ba71b48fdbf72ad27f3446400ebcdb9";
      }}/src/mocha/userstyle.css";
    };
    packages = with pkgs; [
      (ewl "figma" "${figma-linux}/bin/figma-linux")
      (ewl "discord" "${vesktop}/bin/vesktop")
      (ewl "vscode" "${vscodium}/bin/codium")
      (ewl "tetrio" "${tetrio-desktop.override{withTetrioPlus=true;}}/bin/tetrio")
      (ewl "bruno" "${bruno}/bin/bruno")
      (ewl "joplin" "${joplin-desktop}/bin/joplin-desktop")
      (ewl "signal" "${signal-desktop}/bin/signal-desktop")
      (ewl "element" "${element-desktop}/bin/element-desktop")
      (jwl "simplex" "${simplex-chat-desktop}/bin/simplex-chat-desktop")
      (ewl "slack" "${slack}/bin/slack")
      (ewl "chrome" "${ungoogled-chromium}/bin/chromium")
      ungoogled-chromium
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
