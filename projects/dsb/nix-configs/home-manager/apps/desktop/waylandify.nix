{config, pkgs, stablep, ...}:
let
  # base
  base = name: envs: binaryPath: args: (pkgs.writeShellScriptBin "${name}" ''
    env ${envs} ${binaryPath} ${args}
  '');
  # chromium-waylandify
  cwl = name: binaryPath: base name "" binaryPath "--enable-features=UseOzonePlatform,WaylandWindowDecorations --ozone-platform-hint=auto --ozone-platform=wayland --enable-wayland-ime";
  # java-waylandify
  jwl = name: binaryPath: base name "" binaryPath "-Dawt.toolkit.name=WLToolkit";
  # qt-waylandify
  qwl = name: binaryPath: base name "QT_QPA_PLATFORM=wayland" binaryPath "";
  electrons = with pkgs; [
    (cwl "figma" "${figma-linux}/bin/figma-linux")
    (cwl "discord" "${vesktop}/bin/vesktop")
    (cwl "vscode" "${vscodium}/bin/codium")
    (cwl "tetrio" "${tetrio-desktop.override{withTetrioPlus=true;}}/bin/tetrio")
    (cwl "bruno" "${bruno}/bin/bruno")
    (cwl "joplin" "${joplin-desktop}/bin/joplin-desktop")
    (cwl "signal" "${stablep.signal-desktop}/bin/signal-desktop")
    (cwl "element" "${element-desktop}/bin/element-desktop")
    (cwl "slack" "${slack}/bin/slack")
  ];
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
      (jwl "simplex" "${simplex-chat-desktop}/bin/simplex-chat-desktop")
      (cwl "chrome" "${ungoogled-chromium}/bin/chromium")
      (qwl "monero" "${monero-gui}/bin/monero-wallet-gui")
      ungoogled-chromium
    ] ++ electrons;
  };
  programs = {
    joplin-desktop.enable = true;
    vscode = {
      enable = true;
      package = pkgs.vscode-with-extensions.override { vscodeExtensions = with pkgs.vscode-extensions; [ ms-vsliveshare.vsliveshare ]; };
    };
  };
}
