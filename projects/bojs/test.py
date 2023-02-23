from json import loads as jload
from os import getcwd, listdir, _exit
from subprocess import Popen
from bojapi import BaekjoonProb
from tomli import loads
from sys import argv
from misilelibpy import read_once

runs = 1

langlist = [".py", ".rb", ".c", ".cpp"]
a = loads(read_once('test.toml'))
globalvals = a['global']
flist = []
for i in listdir(getcwd()):
    for i2 in langlist:
        if i.endswith(i2):
            flist.append(i.removesuffix(i2))
            break

flist = list(filter(lambda x: x.isnumeric(), flist))
try:
    argv[1]
except IndexError:
    pass
else:
    flist = [argv[1]]
d = []
g = []

for i, i2 in enumerate(flist):
    print(i2)
    args = [f"hyperfine --runs {runs} --export-json test.json --output ./output.txt", "", "< input.txt'"]
    config = a.get(i2, globalvals)
    for i3 in config["lang"]:
        if i3 == "pypy":
            s = f"'pypy {i2}.py"
        elif i3 == "python":
            s = f"'python {i2}.py"
        elif i3 == "ruby":
            s = f"'ruby --enable-yjit {i2}.rb"
        elif i3 == "rustpython":
            s = f"'rustpython {i2}.py"
        elif i3 in ["c", "cpp"]:
            s = f"'clang -O2 {i2}.c -o main && ./main"
        else:
            raise ValueError("no support lang")
        args[1] = s
    bjp = BaekjoonProb(i2)
    print(' '.join(args))
    for inp, output in zip(bjp.sample_input, bjp.sample_output):
        with open("input.txt", "w", encoding="utf8") as file:
            file.write(inp.replace('\r', ''))
        p = Popen(' '.join(args), shell=True)
        _wait = p.wait()
        if _wait != 0:
            print(f"Error Code {_wait}")
            print(f"stdout: {p.stdout}")
            print(f"stderr: {p.stderr}")
            _exit(_wait)
        outp = read_once('output.txt')
        print(outp)
        for i3 in jload(read_once('test.json'))["results"]:
            outpu = output.replace('\r', '')
            if outp not in [outpu.removesuffix('\n'), outpu] and outp.strip("\n") not in [outpu.removesuffix('\n'), outpu]:
                print(f"{i3['command']} does not match with {output} so failed")
                g.append(i3['command'])
            elif i3['times'][0] > bjp.time_limit:
                print(f"{i3['command']} took {i3['times'][0]} seconds, which is more than {bjp.time_limit} seconds, so test failed")
                d.append(i3['command'])
            else:
                print(f"{i3['command']} took {i3['times'][0]} seconds, which is less than {bjp.time_limit} seconds, so test passed")

for i in d:
    print(f"Test failed by time - {i}")

for i in g:
    print(f"Test failed by output - {i}")
