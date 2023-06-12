from os import listdir
from os.path import isdir
from tomli import load
from tomli_w import dump

excludefolder = ["h4", "ap", "xobusy"]
python_version = "3.11.4"

for i in [x for x in listdir('.') if isdir(x)]:
    if i not in excludefolder:
        with open(f"{i}/pyproject.toml", "rb") as f:
            temp = load(f)
            temp["tool"]["poetry"]["dependencies"]["python"] = python_version
            dump(temp, open(f"{i}/pyproject.toml", "wb"))
