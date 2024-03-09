from git.repo import Repo
from sys import argv
from subprocess import run

run("git-lfs install && git-lfs pull", shell=True, check=True)

try:
  a = Repo(argv[1])
except Exception:
  a = Repo()
b = a.head.commit.diff("HEAD^1")
c = [
  i.a_path for i in b 
  if i.change_type == "A" and 
  i.a_path.endswith(".jar") and 
  i.a_path.count("servermods") == 0
]
print(c)
if c:
  run(["7zz", "a", "added.7z", *c], shell=True, check=True)
