from sys import argv

lang = argv[1]
release = argv[2] == "release"

commonc = ["-pipe -O2", "-pipe -g"]

flagc = {
    "c": commonc,
    "cpp": commonc,
    "zig": ["-OReleaseFast", ""],
    "ruby": ["--yjit", "--yjit"]
}

if release:
    print(commonc[0], end="")
else:
    print(commonc[1], end="")
