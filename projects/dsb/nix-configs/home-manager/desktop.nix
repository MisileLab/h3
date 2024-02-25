{ config, pkgs, catppuccin, ... }:
let 
  custom-ctps = {
    waybar = builtins.fetchGit{url="https://github.com/catppuccin/waybar.git";rev="f74ab1eecf2dcaf22569b396eed53b2b2fbe8aff";};
    dunst = builtins.fetchGit{url="https://github.com/catppuccin/dunst.git";rev="a72991e56338289a9fce941b5df9f0509d2cba09";};
  };
  returnColorCSS = {r, g, b, a, addi ? ""}: ''
    /*backdrop-filter: blur(5px);*/
    background: rgba(${toString r}, ${toString g}, ${toString b}, ${toString a});
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    border-radius: 0;
    ${addi}
  '';
in
{
  xdg.enable = true;
  home.pointerCursor = {
    name = "Adwaita";
    package = pkgs.gnome.adwaita-icon-theme;
    size = 32;
  };

  i18n = {
    inputMethod = {
      enabled = "kime";
      kime.config = {
        indicator.icon_color = "White";
      };
    };
    package = pkgs.swayfx; # only for x86_64
  };

  wayland.windowManager.sway = {
    enable = true;
    extraConfigEarly = ''
      blur enable
      blur_xray enable
      blur_radius 10
      corner_radius 8
      blur_passes 4
      shadows enable
      for_window [class=\".*\"] opacity 0.9
      for_window [app_id=\".*\"] opacity 0.9
      bindsym Shift+Print	exec ${pkgs.sway-contrib.grimshot} --notify save area
    '';
    config = {
      bars = [];
      startup = [
        {command = "${pkgs.waybar}/bin/waybar";}
        {command = "${pkgs.swaybg}/bin/swaybg --image ~/.config/home-manager/bg.png";}
        {command = "${pkgs.wl-clipboard}/bin/wl-paste -t text --watch ${pkgs.clipman} store --no-persist";}
      ];
      terminal = "${pkgs.alacritty}/bin/alacritty";
      fonts = {
        names = ["Fira Code NF" "Fira Code" "NanumSquare"];
        style = "Regular";
        size = 12.0;
      };
      window.titlebar = false;
    };
    package = (if pkgs.system == "x86_64-linux" then pkgs.swayfx else pkgs.sway);
    xwayland = true;
  };

  programs = {
    fish = {catppuccin.enable = true;plugins = [{name="tide";src=pkgs.fishPlugins.tide.src;}];};
    alacritty = {
      enable = true;
      catppuccin.enable = true;
      settings = {shell = "${pkgs.fish}/bin/fish";};
    };
    waybar = {
      enable = true;
      settings = [{
        modules-left = [ "sway/workspaces"];
        modules-center = [ "sway/window" ];
        modules-right = [ "pulseaudio" "network" "battery" "cpu" "memory" "clock"];
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
          format-disconnected = "";
          tooltip-format = "{ifname} via {gwaddr} 󰊗";
          tooltip-format-wifi = "{essid} ({signalStrength}%) ";
          tooltip-format-ethernet = "{ifname} ";
          tooltip-format-disconnected = "Disconnected";
          max-length = 50;
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
     virt-manager.enable = true;
     sway = {window = {
       format = "{app_id} - {title}";
       max-length = 20;
    };};
      }];
      style = ''
        @import "${custom-ctps.waybar}/themes/mocha.css";
        * {
          font-family: 'Fira Code', monospace;
        }
        window#waybar {${returnColorCSS({r=108;g=112;b=134;a=0.4;})}}
        #workspaces button {
          ${returnColorCSS({r=108;g=112;b=134;a=0.4;addi=''
            padding-left: 6px;
            padding-right: 6px;
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
            padding-left: 6px;
            padding-right: 6px;
          '';})}
        }
        .modules-left * {
          color: @subtext1;
        }
        .modules-center * {
          color: @text;
        }
        .modules-right * {
          ${returnColorCSS({r=147;g=153;b=178;a=0.6;addi=''
            padding-left: 6px;
            padding-right: 6px;
            color: @subtext1;
          '';})};
        }
        #battery.warning {
          color: @yellow;
        }
        #battery.critical {
          color: @red;
        }
      '';
    };
    firefox.enable = true;
    rofi = {enable = true;extraConfig = {
#configuration = {
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
#};
};
theme = "catppuccin-mocha";
};
};

  services = { dunst = {enable = true; configFile = "${custom-ctps.dunst}/src/mocha.conf";};};
}
