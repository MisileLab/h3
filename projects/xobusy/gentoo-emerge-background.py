from sys import argv
from os import _exit, chdir, getcwd
from os.path import expanduser
from subprocess import run

try:
    argv[1]
    if argv[1] == "command":
        argv[2]
except IndexError:
    print("Use: <gentoo emerge background file> <viewlog/command>")
    _exit(0)

if argv[1] == "command":
    _example = argv
    del _example[1]
    _org = getcwd()
    chdir(expanduser("~"))
    run(f'nohup {" ".join(_example)} &')
    chdir(_org)
elif argv[1] == "viewlog":
    run(f'tail -f {expanduser("~")}/nohup.out')
else:
    print("Use: <gentoo emerge background file> <viewlog/command>")
