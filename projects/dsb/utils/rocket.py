from subprocess import run
from os import get_terminal_size
from time import sleep

print(f"{chr(27)}[2J")
outs = run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.removesuffix("\n")
rocket = rf"""
          !
          !
          ^
         / \
        /   \
       /     \
      /       \
     /         \
    /___________\
    |=         =|
    | {outs} |
    |           |
    |           |
    |           |
    |           |
    |           |
    |           |
    |           |
    |           |
   /|#####!#####|\
  / |#####!#####| \
 /  |#####!#####|  \
|  /   ^  |  ^  \  |
| /    (  |  )   \ |
|/     (  |  )    \|
"""  # noqa: E501

for i in range(get_terminal_size().lines):
  print(rocket + "\n" * i)
  sleep(1)
  print(f"{chr(27)}[2J")
