import inquirer
from subprocess import run

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

output = answers["type"]

if answers["cscope"] == "y":
  output = f"{output}({answers["cscopecon"]})"

if answers["bchanges"] == "y":
  output = f"{output}!"

output = f"{output}: {answers["shortdesc"]}"

if len(answers["longdesc"]) != 0:
  output = f"{output}\n\n{answers["longdesc"]}"

if answers["bchanges"] == "y":
  output = f"{output}\n\n{answers["bchangescon"]}"

print(output)

