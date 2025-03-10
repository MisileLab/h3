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
  electrons = with pkgs; [
    (cwl "figma" "${figma-linux}/bin/figma-linux")
    (cwl "discord" "${vesktop}/bin/vesktop")
    (cwl "vscode" "${vscodium}/bin/codium")
  ];
  chromiums = with stablep; [
    (cwl "chrome" "${ungoogled-chromium}/bin/chromium")
    ungoogled-chromium
  ];
in
{
  home = {
    packages = with pkgs; [
      (jwl "simplex" "${simplex-chat-desktop}/bin/simplex-chat-desktop")
      (qwl "monero" "${monero-gui}/bin/monero-wallet-gui")
    ] ++ electrons ++ chromiums;
    file = {
      ".config/simplex/catppuccin-mocha.theme".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/simplex";
        rev="4166f14ec29d4a5d863f095e259512bdf32d2556";
      }}/themes/catppuccin-mocha.theme";
    };
  };
}
