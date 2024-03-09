from os.path import isdir, isfile, getsize, join, islink
from os import _exit, walk
from sys import argv

a = 0

def sizeof_fmt(num, suffix="B"):
  for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
    if abs(num) < 1024.0:
      return f"{num:3.1f}{unit}{suffix}"
    num /= 1024.0
  return f"{num:.1f}Yi{suffix}"

def get_folder_size(start_path = '.'):
  total_size = 0
  for dirpath, _, filenames in walk(start_path):
    for f in filenames:
      fp = join(dirpath, f)
      if not islink(fp):
        total_size += getsize(fp)

  return total_size

try:
  argv[1]
except KeyError:
  print("no arg")
  _exit(1)

del argv[0]

for i in argv:
  if isdir(i):
    a += get_folder_size(i)
  elif isfile(i):
    a += getsize(i)
  else:
    print("No folder or file so invaild")
    _exit(1)

print(sizeof_fmt(a))
