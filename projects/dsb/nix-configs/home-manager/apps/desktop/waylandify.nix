{pkgs, stablep, config, ...}:
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
  electrons = with stablep; [
    (cwl "figma" "${figma-linux}/bin/figma-linux")
    (cwl "discord" "${vesktop}/bin/vesktop")
    (cwl "vscode" "${vscodium}/bin/codium")
  ];
in
{
  home = {
    packages = with pkgs; [
      (jwl "simplex" "${simplex-chat-desktop}/bin/simplex-chat-desktop")
      (cwl "chrome" "${stablep.ungoogled-chromium}/bin/chromium")
      (qwl "monero" "${monero-gui}/bin/monero-wallet-gui")
      stablep.ungoogled-chromium
    ] ++ electrons;
    file = {
      ".config/simplex/catppuccin-mocha.theme".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/simplex";
        rev="4166f14ec29d4a5d863f095e259512bdf32d2556";
      }}/themes/catppuccin-mocha.theme";
    };
  };
  programs = {
    vscode = {
      enable = true;
      package = pkgs.vscode-with-extensions.override { vscodeExtensions = with pkgs.vscode-extensions; [ ms-vsliveshare.vsliveshare ]; };
    };
  };
}
