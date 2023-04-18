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
    del _example[0]
    del _example[0]
    print(_example)
    _org = getcwd()
    chdir(expanduser("~"))
    # file deepcode ignore Python/CommandInjection: <please specify a reason of ignoring this>, file deepcode ignore CommandInjection: <please specify a reason of ignoring this>
    run(f'nohup {" ".join(_example)} &', shell=True)
    chdir(_org)
elif argv[1] == "viewlog":
    run(f'tail -f {expanduser("~")}/nohup.out', shell=True)
else:
    print("Use: <gentoo emerge background file> <viewlog/command>")
