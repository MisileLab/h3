{pkgs, ...}: {
  programs.niri = {
    enable = true;
    package = pkgs.niri;
    settings = {
      outputs."eDP-1".scale = 2.0;
      window-rules = [
        {
          matches = [
            {app-id = "GalaxyBudsClient";}
            {app-id = "org.gnome.Loupe";}
          ];
          open-floating = true;
        }
      ];
      prefer-no-csd = true;
      spawn-at-startup = [
        { command = ["${pkgs.xwayland-satellite}/bin/xwayland-satellite"]; }
        { command = ["${pkgs.waybar}/bin/waybar"]; }
        { command = ["${pkgs.swaybg}/bin/swaybg" "--image" "~/.config/home-manager/bg.png"]; }
        { command = ["${pkgs.wl-clipboard}/bin/wl-paste" "--watch" "${pkgs.cliphist}/bin/cliphist" "store"]; }
        { command = ["${pkgs.avizo}/bin/avizo-service"]; }
        { command = [
          "${pkgs.swayidle}/bin/swayidle"
          "-w"
          "timeout" "601" "niri msg action power-off-monitors"
          "timeout" "600" "swaylock -f"
          "before-sleep" "swaylock -f"
        ]; }
      ];
      binds = {
        "Mod+Shift+Slash".action.show-hotkey-overlay = {};

        "Mod+T".action.spawn = ["${pkgs.ghostty}/bin/ghostty"];
        "Mod+D".action.spawn = ["${pkgs.rofi-wayland}/bin/rofi" "-show" "drun"];
        "Mod+Y".action.spawn = ["sh" "-c" "${pkgs.cliphist}/bin/cliphist list | ${pkgs.rofi-wayland}/bin/rofi -dmenu | ${pkgs.cliphist}/bin/cliphist decode | ${pkgs.wl-clipboard}/bin/wl-copy"];
        "Mod+Shift+Y".action.spawn = ["sh" "-c" "${pkgs.cliphist}/bin/cliphist list | ${pkgs.rofi-wayland}/bin/rofi -dmenu | ${pkgs.cliphist}/bin/cliphist delete"];
        "Ctrl+Shift+Y".action.spawn = ["${pkgs.cliphist}/bin/cliphist" "wipe"];
        "Super+Alt+L".action.spawn = "${pkgs.swaylock}/bin/swaylock";

        "XF86AudioRaiseVolume" = {
          allow-when-locked = true;
          action.spawn = ["${pkgs.avizo}/bin/volumectl" "-u" "up"];
        };
        "XF86AudioLowerVolume" = {
          allow-when-locked = true;
          action.spawn = ["${pkgs.avizo}/bin/volumectl" "-u" "down"];
        };
        "XF86AudioMute" = {
          allow-when-locked = true;
          action.spawn = ["${pkgs.avizo}/bin/volumectl" "toggle-mute"];
        };
        "XF86AudioMicMute" = {
          allow-when-locked = true;
          action.spawn = ["${pkgs.avizo}/bin/volumectl" "-m" "toggle-mute"];
        };

        "Mod+Q".action.close-window = {};

        "Mod+Left".action.focus-column-left = {};
        "Mod+Down".action.focus-window-down = {};
        "Mod+Up".action.focus-window-up = {};
        "Mod+Right".action.focus-column-right = {};
        "Mod+H".action.focus-column-left = {};
        "Mod+J".action.focus-window-down = {};
        "Mod+K".action.focus-window-up = {};
        "Mod+L".action.focus-column-right = {};

        "Mod+Shift+Left".action.move-column-left = {};
        "Mod+Shift+Down".action.move-window-down = {};
        "Mod+Shift+Up".action.move-window-up = {};
        "Mod+Shift+Right".action.move-column-right = {};
        "Mod+Shift+H".action.move-column-left = {};
        "Mod+Shift+J".action.move-window-down = {};
        "Mod+Shift+K".action.move-window-up = {};
        "Mod+Shift+L".action.move-column-right = {};

        "Mod+Home".action.focus-column-first = {};
        "Mod+End".action.focus-column-last = {};
        "Mod+Ctrl+Home".action.move-column-to-first = {};
        "Mod+Ctrl+End".action.move-column-to-last = {};

        "Mod+Ctrl+Left".action.focus-monitor-left = {};
        "Mod+Ctrl+Down".action.focus-monitor-down = {};
        "Mod+Ctrl+Up".action.focus-monitor-up = {};
        "Mod+Ctrl+Right".action.focus-monitor-right = {};
        "Mod+Ctrl+H".action.focus-monitor-left = {};
        "Mod+Ctrl+J".action.focus-monitor-down = {};
        "Mod+Ctrl+K".action.focus-monitor-up = {};
        "Mod+Ctrl+L".action.focus-monitor-right = {};

        "Mod+Shift+Ctrl+Left".action.move-column-to-monitor-left = {};
        "Mod+Shift+Ctrl+Down".action.move-column-to-monitor-down = {};
        "Mod+Shift+Ctrl+Up".action.move-column-to-monitor-up = {};
        "Mod+Shift+Ctrl+Right".action.move-column-to-monitor-right = {};
        "Mod+Shift+Ctrl+H".action.move-column-to-monitor-left = {};
        "Mod+Shift+Ctrl+J".action.move-column-to-monitor-down = {};
        "Mod+Shift+Ctrl+K".action.move-column-to-monitor-up = {};
        "Mod+Shift+Ctrl+L".action.move-column-to-monitor-right = {};

        "Mod+Page_Down".action.focus-workspace-down = {};
        "Mod+Page_Up".action.focus-workspace-up = {};
        "Mod+U".action.focus-workspace-down = {};
        "Mod+I".action.focus-workspace-up = {};
        "Mod+Ctrl+Page_Down".action.move-column-to-workspace-down = {};
        "Mod+Ctrl+Page_Up".action.move-column-to-workspace-up = {};
        "Mod+Ctrl+U".action.move-column-to-workspace-down = {};
        "Mod+Ctrl+I".action.move-column-to-workspace-up = {};

        "Mod+Shift+Page_Up".action.move-workspace-up = {};
        "Mod+Shift+U".action.move-workspace-down = {};
        "Mod+Shift+I".action.move-workspace-up = {};

        "Mod+WheelScrollDown" = {
          cooldown-ms = 150;
          action.focus-workspace-down = {};
        };
        "Mod+WheelScrollUp" = {
          cooldown-ms = 150;
          action.focus-workspace-up = {};
        };
        "Mod+Ctrl+WheelScrollDown" = {
          cooldown-ms = 150;
          action.move-column-to-workspace-down = {};
        };
        "Mod+Ctrl+WheelScrollUp" = {
          cooldown-ms = 150;
          action.move-column-to-workspace-up = {};
        };

        "Mod+WheelScrollRight".action.focus-column-right = {};
        "Mod+WheelScrollLeft".action.focus-column-left = {};
        "Mod+Ctrl+WheelScrollRight".action.move-column-right = {};
        "Mod+Ctrl+WheelScrollLeft".action.move-column-left = {};

        # Usually scrolling up and down with Shift in applications results in
        # horizontal scrolling; these binds replicate that.
        "Mod+Shift+WheelScrollDown".action.focus-column-right = {};
        "Mod+Shift+WheelScrollUp".action.focus-column-left = {};
        "Mod+Ctrl+Shift+WheelScrollDown".action.move-column-right = {};
        "Mod+Ctrl+Shift+WheelScrollUp".action.move-column-left = {};

        "Mod+1".action.focus-workspace = 1;
        "Mod+2".action.focus-workspace = 2;
        "Mod+3".action.focus-workspace = 3;
        "Mod+4".action.focus-workspace = 4;
        "Mod+5".action.focus-workspace = 5;
        "Mod+6".action.focus-workspace = 6;
        "Mod+7".action.focus-workspace = 7;
        "Mod+8".action.focus-workspace = 8;
        "Mod+9".action.focus-workspace = 9;
        "Mod+Shift+1".action.move-column-to-workspace = 1;
        "Mod+Shift+2".action.move-column-to-workspace = 2;
        "Mod+Shift+3".action.move-column-to-workspace = 3;
        "Mod+Shift+4".action.move-column-to-workspace = 4;
        "Mod+Shift+5".action.move-column-to-workspace = 5;
        "Mod+Shift+6".action.move-column-to-workspace = 6;
        "Mod+Shift+7".action.move-column-to-workspace = 7;
        "Mod+Shift+8".action.move-column-to-workspace = 8;
        "Mod+Shift+9".action.move-column-to-workspace = 9;

        "Mod+Comma".action.consume-window-into-column = {};
        "Mod+Period".action.expel-window-from-column = {};

        "Mod+R".action.switch-preset-column-width = {};
        "Mod+Shift+R".action.reset-window-height = {};
        "Mod+F".action.maximize-column = {};
        "Mod+Shift+F".action.fullscreen-window = {};
        "Mod+C".action.center-column = {};

        "Mod+Minus".action.set-column-width = "-10%";
        "Mod+Equal".action.set-column-width = "+10%";

        # Finer height adjustments when in column with other windows.
        "Mod+Shift+Minus".action.set-window-height = "-10%";
        "Mod+Shift+Equal".action.set-window-height = "+10%";

        "Print".action.screenshot = {};
        "Ctrl+Print".action.screenshot-screen = {};
        "Alt+Print".action.screenshot-window = {};

        # The quit action will show a confirmation dialog to avoid accidental exits.
        "Mod+Shift+E".action.quit = {};

        # Powers off the monitors. To turn them back on, do any input like
        # moving the mouse or pressing any other key.
        "Mod+Shift+P".action.power-off-monitors = {};
      };
    };
  };
}
