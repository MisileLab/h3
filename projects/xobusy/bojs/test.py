from os import getcwd, listdir
from subprocess import Popen, run
from json import loads as jload
from baekjoonapi import BaekjoonProb
from tomli import loads
from sys import argv
from difflib import SequenceMatcher
from pathlib import Path

runs = 1


def read_once(a: str) -> str:
  return Path(a).read_text()


def diff_strings(a: str, b: str, *, use_loguru_colors: bool = False) -> str:
  output = []
  matcher = SequenceMatcher(None, a, b)
  if use_loguru_colors:
    green = '<GREEN><black>'
    red = '<RED><black>'
    endgreen = '</black></GREEN>'
    endred = '</black></RED>'
  else:
    green = '\x1b[38;5;16;48;5;2m'
    red = '\x1b[38;5;16;48;5;1m'
    endgreen = '\x1b[0m'
    endred = '\x1b[0m'

  for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
    if opcode == 'delete':
      output.append(f'{red}{a[a0:a1]}{endred}')
    elif opcode == 'equal':
      output.append(a[a0:a1])
    elif opcode == 'insert':
      output.append(f'{green}{b[b0:b1]}{endgreen}')
    elif opcode == 'replace':
      output.extend(
        (f'{green}{b[b0:b1]}{endgreen}', f'{red}{a[a0:a1]}{endred}'))
  return ''.join(output)


langlist = [".py", ".rb", ".c", ".cpp"]
probtypes = ["boj"]
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


class UniversalProblemFormat:

  def __init__(self, sample_input: list, sample_output: list,
         time_limit: float):
    self.sample_input = sample_input
    self.sample_output = sample_output
    self.time_limit = time_limit


def pb_info(prob: str, name: str) -> UniversalProblemFormat:
  # sourcery skip: extract-method
  if prob not in probtypes:
    raise NotImplementedError("no support prob")
  if prob == "boj":
    bjp = BaekjoonProb(name)
    dup = a.get(name, {"sample_input": [], "sample_output": []})
    sinput, soutput = bjp.sample_input, bjp.sample_output
    sinput.extend(dup.get("sample_input", []))
    soutput.extend(dup.get("sample_output", []))
    return UniversalProblemFormat(sinput, soutput, bjp.time_limit)
  raise NotImplementedError("Unreachable")


for i, i2 in enumerate(flist):
  print(i2)
  args = [
    f"hyperfine --runs {runs} --export-json test.json --output ./output.txt",
    "", "< input.txt'"
  ]
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
    elif i3 == "c":
      s = f"'clang -O2 {i2}.c -o main && ./main"
    elif i3 == "cpp":
      s = f"'clang++ -O2 {i2}.c -o main && ./main"
    else:
      raise ValueError("no support lang")
    args[1] = s
  bjp = pb_info("boj", i2)
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
      exit(_wait)
    outp = read_once('output.txt')
    print(outp)
    for i3 in jload(read_once('test.json'))["results"]:
      outpu = output.replace('\r', '')
      if outp not in [outpu.removesuffix('\n'), outpu
              ] and outp.strip("\n") not in [
                outpu.removesuffix('\n'), outpu
              ]:  # noqa: E501
        print(
          f"{i3['command']} does not match with {output} so failed")
        print(diff_strings(outp, outpu))
        exit(1)
      elif i3['times'][0] > bjp.time_limit:
        print(
          f"{i3['command']} took {i3['times'][0]} seconds, which is more than {bjp.time_limit} seconds, so test failed"
        )  # noqa: E501
        exit(1)
      else:
        print(
          f"{i3['command']} took {i3['times'][0]} seconds, which is less than {bjp.time_limit} seconds, so test passed"
        )  # noqa: E501

if len(flist) != 1:
  exit(0)

ans = ""
while ans not in ["y", "n"]:
  ans = input("do u want some copy? (needs wl-copy) [y/n]")
if ans == "y":
  print("=======")
  run(f"cat {flist[0]}.py | wl-copy", shell=True)
