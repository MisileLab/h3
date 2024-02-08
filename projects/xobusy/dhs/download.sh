#!/usr/bin/env fish

wget $argv[1] -O a.zip
ouch decompress a.zip
rm a.zip

