from os import listdir
from pathlib import Path

for i in listdir("queries"):
  modules: list[str] = []
  print(i)
  for j in listdir(f"queries/{i}"):
    if not j.endswith(".py") or j == "__init__.py":
      continue
    print(j)
    modules.append(j.removesuffix("_async_edgeql.py"))
  _ = Path(f"queries/{i}/__init__.py").write_text(
    "\n".join(f"from .{j}_async_edgeql import {j}" for j in modules)
    + f"\n__all__ = [{', '.join(f"'{j}'" for j in modules)}]"
  )
