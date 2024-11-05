import inquirer

from subprocess import run
from sys import argv
from json import loads, dumps

j = input("Enter previous commit json(if you don't have, just enter): ")

if j == "":
  convtypes = ["feat","fix","docs","style","refactor","perf","test","build","ci","chore","revert"]

  questions = [
    inquirer.List("type", message="Type of commit", choices=convtypes),
    inquirer.List("cscope", message="Custom scope", choices=["y","n"]),
    inquirer.Text("cscopecon", message="Content of custom scope", ignore=lambda x: x["cscope"] == "n"),
    inquirer.Text("shortdesc", message="Short description(must be less than 55 chars)", validate=lambda _, x: len(x) <= 55),
    inquirer.Editor("longdesc", message="Long description"),
    inquirer.List("bchanges", message="Breaking changes", choices=["y","n"]),
    inquirer.Editor("bchangescon", message="Breaking change content", ignore=lambda x: x["bchanges"] == "n")
  ]

  answers = inquirer.prompt(questions)

  if answers is None:
    exit(1)

  print(dumps(answers))
else:
  answers = loads(j)

output = answers["type"]

if answers["cscope"] == "y":
  output = f"{output}({answers["cscopecon"]})"

if answers["bchanges"] == "y":
  output = f"{output}!"

output = f"{output}: {answers["shortdesc"]}"

if answers["longdesc"].strip("\n") != "":
  output = f"{output}\n{answers["longdesc"]}"

if answers["bchanges"] == "y":
  output = f"{output}\n--breaking changes--\n{answers["bchangescon"]}"

print(output)
answer = inquirer.prompt([inquirer.Confirm('confirm', message='Is it right?')])
if answer is not None and answer.get('confirm', None) is True:
  add_all = inquirer.prompt([inquirer.Confirm('confirm', message='Do you want add all before commit?')])
  if add_all is not None and add_all.get('confirm', None) is True:
    run(args=["git", "add", "-A"])
  push = inquirer.prompt([inquirer.Confirm('confirm', message='Do you want to push it?')])
  run(args=["git", "commit", "-m", output, "-s"])
  if push is not None and push.get('confirm', None) is True:
    run(args=["git", "push"])

