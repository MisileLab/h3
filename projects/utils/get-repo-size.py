from sys import argv
from pathlib import Path
from requests import get

def convert_size(size_in_bytes, precision=2):
    for unit in ['KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB', 'RB', 'QB']:
        if size_in_bytes < 1024:
            return round(size_in_bytes, precision), unit
        size_in_bytes /= 1024
    return round(size_in_bytes, precision), 'QB'

token = Path("token.txt").read_text()
username, reponame = (argv[1], argv[2])

size, sizeext = convert_size(get(f"https://api.github.com/repos/{username}/{reponame}").json()["size"])
print(f"size = {size}{sizeext}")
