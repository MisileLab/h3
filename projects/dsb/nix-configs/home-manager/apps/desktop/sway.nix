{c, pkgs, ...}:
let
  returnColorCSS = {r, g, b, a, addi ? ""}: ''
    ${(if c.gtk4 then "backdrop-filter: blur(5px)" else "")}
    background: rgba(${toString r}, ${toString g}, ${toString b}, ${toString a});
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    border-radius: 0;
    ${addi}
  '';
in
{
  home.packages = with pkgs; [swaysome swayimg jq libnotify];
  wayland.windowManager.sway = {
    enable = true;
    extraConfigEarly = ''
seat * shortcuts_inhibitor disable

input type:touchpad {
  tap enabled
  natural_scroll enabled
}

blur enable
blur_xray enable
blur_radius 10
blur_passes 4
shadows enable

for_window [class=".*"] opacity 0.9
for_window [app_id=".*"] opacity 0.9
for_window [instance="GalaxyBudsClient"] floating enable

bindsym Print exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify copy area
bindsym Shift+Print	exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify copy screen
bindsym Shift+Alt+Print exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify savecopy anything
bindsym Alt+Print exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify save area

bindsym XF86AudioRaiseVolume exec ${pkgs.avizo}/bin/volumectl -u up
bindsym XF86AudioLowerVolume exec ${pkgs.avizo}/bin/volumectl -u down
bindsym XF86AudioMute exec ${pkgs.avizo}/bin/volumectl toggle-mute
bindsym XF86AudioMicMute exec ${pkgs.avizo}/bin/volumectl -m toggle-mute
bindsym XF86MonBrightnessUp exec ${pkgs.avizo}/bin/lightctl up
bindsym XF86MonBrightnessDown exec ${pkgs.avizo}/bin/lightctl down

bindsym Mod4+y exec ${pkgs.clipman}/bin/clipman pick --tool="rofi" --max-items=30
bindsym Mod4+shift+y exec ${pkgs.swayfx}/bin/swaynag --type warning -m 'You want to clear clipboard?' -b 'Yes' 'exec ${pkgs.clipman}/bin/clipman clear --all'
    '';
    extraConfig = ''
titlebar_separator disable
hide_edge_borders --i3 smart_no_gaps
font pango:monospace 0.001
titlebar_border_thickness 0
titlebar_padding 1
    '';
    config = {
      bars = [];
      startup = [
        {command = 
          ''
          ${pkgs.swayidle}/bin/swayidle -w \
              timeout 60 '${pkgs.swaylock}/bin/swaylock -f' \
              timeout 90 "${pkgs.swayfx}/bin/swaymsg 'output * dpms off'" \
              resume "${pkgs.swayfx}/bin/swaymsg 'output * dpms on'" \
              before-sleep '${pkgs.swaylock}/bin/swaylock -f' \
              lock '${pkgs.swaylock}/bin/swaylock -f'
          '';}
        {command = "${pkgs.waybar}/bin/waybar";}
        {command = "${pkgs.swaybg}/bin/swaybg --image ~/.config/home-manager/bg.png";}
        {command = "${pkgs.wl-clipboard}/bin/wl-paste -t text --watch ${pkgs.clipman}/bin/clipman store --no-persist";}
        {command = "${pkgs.avizo}/bin/avizo-service";}
      ];
      menu = "${pkgs.rofi-wayland}/bin/rofi -show drun";
      terminal = "${pkgs.alacritty}/bin/alacritty";
      fonts = {
        names = ["FiraCode Nerd Font Mono" "NanumSquare"];
        style = "Regular";
        size = 12.0;
      };
      window = {
        titlebar = false;
        border = 0;
      };
      modifier = "Mod4";
    };
    checkConfig = false;
    package = pkgs.swayfx;
    xwayland = true;
  };
  programs = {
    rofi = {
      enable = true;
      catppuccin.enable = true;
      extraConfig = {
        modi = "run,drun,window";
        icon-theme = "Oranchelo";
        show-icons = true;
        terminal = "alacritty";
        drun-display-format = "{icon} {name}";
        location = 0;
        hide-scrollbar = true;
        display-drun = "   Apps ";
        display-run = "   Run ";
        display-window = " 﩯  Window";
        disable-history = false;
        display-Network = " 󰤨  Network";
        sidebar-mode = true;
      };
      package = pkgs.rofi-wayland;
    };
    swaylock = {
      enable = true;
      catppuccin.enable = true;
    };
    waybar = {
      enable = true;
      catppuccin.enable = true;
      settings = [{
        modules-left = [ "sway/workspaces" ];
        modules-center = [ "sway/window" ];
        modules-right = ["idle_inhibitor" "pulseaudio" "battery" "cpu" "memory" "custom/gpu-usage" "temperature" "clock"];
        "idle_inhibitor" = {
          "format" = "{icon}";
          "format-icons" = {activated = "󰅶"; deactivated = "󰾪";};
          "on-click" = "$HOME/.config/home-manager/idle-handler.sh";
        };
        "custom/gpu-usage" = {
          exec = "cat /sys/class/hwmon/hwmon0/device/gpu_busy_percent";
          format = " {}%";
          return-type = "";
          interval = 1;
        };
        backlight = {
          device = "intel_backlight";
          format = "{icon} {percent}%";
          format-icons = ["" ""];
        };
        cpu = {
          interval = 5;
          format = " {}%";
          max-length = 10;
        };
        memory = {
          interval = 5;
          format = " {}%";
          tooltip-format = "{used}G/{total}G";
          max-length = 10;
      };
        network = {
          interface = "wlan0";
          format = "{ifname}";
          format-wifi = "  {signalStrength}%";
          format-ethernet = "󰊗  {ipaddr}";
          format-disconnected = "Disconnected";
          tooltip-format = "{ifname} via {gwaddr} 󰊗";
          tooltip-format-wifi = "{essid} ({signalStrength}%) ";
          tooltip-format-ethernet = "{ifname} ";
          tooltip-format-disconnected = "Disconnected";
          max-length = 40;
      };
      clock = {
        interval = 1;
        format = "{:%H:%M:%S}";
        tooltip-format = "{:%Y-%m-%dT%H:%M:%S}";
        max-length = 25;
      };
      battery = {
        bat = "BAT1";
        interval = 60;
        states = {
          warning = 30;
          critical = 15;
        };
        format = "{icon} {capacity}%";
        format-icons = [" " " " " " " " " "];
        max-length = 25;
      };
      "sway/window" = {
        format = "{app_id} - {title}";
        max-length = 50;
      };
    }];
    style = ''
      * {
        font-family: 'FiraCode Nerd Font Mono', monospace;
      }
      window#waybar {${returnColorCSS({r=108;g=112;b=134;a=0.4;})}}
      #workspaces button {
        ${returnColorCSS({r=108;g=112;b=134;a=0.4;addi=''
        padding-left: 2px;
        padding-right: 2px;
        '';})}
      }
      #workspaces button.active {
        ${returnColorCSS({r=147;g=153;b=178;a=0.6;})}
      }
      #workspace button.urgent {
        background: rgba(249, 226, 175, 0.6);
      }
      #workspace button.hover {
        ${returnColorCSS({r=127;g=132;b=156;a=0.6;addi=''
        padding-left: 2px;
        padding-right: 2px;
        '';})}
      }
      .modules-left * {
        color: @subtext1;
        padding-left: 0px;
        padding-right: 0px;
      }
      .modules-center * {
        color: @text;
      }
      .modules-right * {
        ${returnColorCSS({r=147;g=153;b=178;a=0.6;addi=''
          padding-left: 6px;
          padding-right: 6px;
          color: @subtext1;
        '';})}
      }
      #battery.warning {
        color: @yellow;
      }
      #battery.critical {
        color: @red;
      }'';
    };
  };
  services = {
    dunst = {
      enable = true;
      catppuccin.enable = true;
    };
    avizo.enable = true;
    poweralertd.enable = true;
  };
}
