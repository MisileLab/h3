#!/usr/bin/env bash

# Check the current idle_inhibit status
STATUS="0"

if [ -f /tmp/idle_status ]; then
  STATUS=`cat /tmp/idle_status`
fi

if [ "$STATUS" == "0" ]; then
    # Enable idle inhibit
    swaymsg "inhibit_idle visible"
    notify-send "Idle Inhibit Enabled"
    echo "1" > /tmp/idle_status
else
    # Disable idle inhibit
    swaymsg "inhibit_idle none"
    notify-send "Idle Inhibit Disabled"
    echo "0" > /tmp/idle_status
fi

