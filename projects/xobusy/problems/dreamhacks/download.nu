#!/usr/bin/env nu
def main [url: string] {
  wget $url -O a.zip
  ouch d a.zip
  rm a.zip
}
