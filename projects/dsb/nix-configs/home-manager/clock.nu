#!/usr/bin/env nu

# Function to get the current time
def get_time [local: bool, formatted: bool = true] {
  if $local {
    let output = date now
    if $formatted {
      $output | format date "%H:%M:%S"
    } else {
      $output
    }
  } else {
    let output = date now | date to-timezone "UTC"
    if $formatted {
      $output | format date "%H:%M:%S"
    } else {
      $output
    }
  }
}

def main [
  local: bool = false,  # Use local time instead of UTC
  copy: string = ""     # Copy to clipboard: "copy" or "copyf" for formatted
] {
  if $copy == "" {
    loop {
      # Just print the time
      let value: datetime = get_time $local;
      let output = {
        text: ($value | format date "%H:%M:%S")
        alt: ($value | format date "%H:%M:%S")
        tooltip: ($value | format date "%Y-%m-%dT%H:%M:%S")
        class: ""
        percentage: 0
      } | to json -r;
      print $output;
      sleep 1sec;
    }
  } else {
    # Handle clipboard copying
    if $copy == "copyf" {
      get_time $local false | format date "%+" | wl-copy
    } else {
      get_time false false | into int | into string | wl-copy
    }
  }
}

