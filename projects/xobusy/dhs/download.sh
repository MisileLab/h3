#!/usr/bin/env fish

if test -d ./a
  echo "delete a"
  rm -rv a
end
wget $argv[1] -O a.zip
ouch decompress a.zip --dir a
rm a.zip

