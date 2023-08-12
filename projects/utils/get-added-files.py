from git.repo import Repo
from sys import argv
from subprocess import run

run("git-lfs install && git-lfs pull", shell=True, check=True)

try:
    a = Repo(argv[1])
except Exception:
    a = Repo()
b = a.head.commit.diff("HEAD^1")
c = [i.a_path for i in b if i.changed_type == "A" and i.a_path.endswith(".jar")]
print(c)
if c is not None:
    run(["7zz", "a", "added.zip", *c], shell=True, check=True)
