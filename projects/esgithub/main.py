from github.Auth import Token
from typer import Typer
from github import Github
from numpy import argmax, array
from tqdm import tqdm
from platformdirs import user_config_dir
from tomli import loads
from openai import OpenAI

from pathlib import Path
from pickle import loads as p_loads, dumps
from collections import defaultdict

app = Typer()
config = loads(Path(user_config_dir("esgithub")).joinpath("config.toml").read_text())
embed_config: dict[str, str] = config.get("embedding", {})
if embed_config.get("provider", "huggingface") == "openai":
  o = OpenAI(api_key=embed_config["api_key"])
  def embed_query(text: str) -> list[float]:
    return o.embeddings.create(
      input=text,
      model=embed_config.get("model", "text-embedding-3-large")
    ).data[0].embedding
else:
  raise ValueError("Invalid provider.")

def template(title: str, description: str):
  return f"{title}\n\n{description}"

p = Path("db.pkl")
data: dict[str, tuple[list[list[float]], list[int]]] = defaultdict(
  lambda: ([], []),
  p_loads(p.read_bytes()) if p.exists() else {}
)

client: Github
if config.get("github", {}).get("token", "") == "": # pyright: ignore[reportAny]
  client = Github()
else:
  client = Github(auth=Token(config["github"]["token"])) # pyright: ignore[reportAny]

@app.command()
def build(repository: str):
  repo = client.get_repo(repository)
  issues = tqdm(repo.get_issues())
  pull_requests = tqdm(repo.get_pulls())
  for i in issues:
    if i.number in data[repository][1]:
      idx = data[repository][1].index(i.number)
      data[repository][0][idx] = embed_query(template(i.title, i.body))
    data[repository][0].append(
      embed_query(template(i.title, i.body))
    )
    data[repository][1].append(i.number)
  for i in pull_requests:
    if i.number in data[repository][1]:
      idx = data[repository][1].index(i.number)
      data[repository][0][idx] = embed_query(template(i.title, i.body))
    data[repository][0].append(
      embed_query(template(i.title, i.body))
    )
    data[repository][1].append(i.number)
  _ = Path("db.pkl").write_bytes(dumps(dict(data)))

@app.command()
def search(repository: str, keyword: str):
  print(keyword)
  if repository not in data:
    print("Repository not found. Please build the database first.")
    return
  embedding = embed_query(keyword)
  distances = array([array(embedding).dot(array(i)) for i in tqdm(data[repository][0])])
  max_index = argmax(distances)
  issue = client.get_repo(repository).get_issue(data[repository][1][max_index])
  print(f"{issue.number}: {issue.title} - {issue.html_url}")

if __name__ == "__main__":
  app()

