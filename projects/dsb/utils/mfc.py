from sys import argv

# Misile's Flag Compiler embedded (MFC)

lang = argv[1]
release = argv[2] == "release"

commonc = ["-pipe -O2", "-pipe -g"]

flagc = {
  "c": commonc,
  "cpp": commonc,
  "zig": ["-Doptimise=ReleaseSafe", ""],
  "ruby": ["--yjit", "--yjit"]
}

if release:
  print(flagc[lang][0], end="")
else:
  print(flagc[lang][1], end="")
