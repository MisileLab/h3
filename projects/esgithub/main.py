from typer import Typer
from embed_anything import EmbeddingModel, WhichModel, embed_query
from github import Github
from numpy import array
from tqdm import tqdm

from pathlib import Path
from pickle import loads, dumps
from collections import defaultdict
from os import getenv

app = Typer()
model: EmbeddingModel = EmbeddingModel.from_pretrained_cloud( # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]
  WhichModel.OpenAI,
  "text-embedding-3-large",  # TODO: embedding configuration
  api_key = getenv("OPENAI_KEY")
) # TODO: support local models

def template(title: str, description: str):
  return f"#title\n{title}\n#description\n{description}"

p = Path("db.pkl")
data: dict[str, tuple[list[list[float]], list[int]]] = loads(
  p.read_bytes()
  ) if p.exists() else defaultdict(lambda: ([], []))

# TODO: path configuration

client = Github()

@app.command()
def build(repository: str):
  repo = client.get_repo(repository)
  issues = list(repo.get_issues())
  pull_requests = list(repo.get_pulls())
  for i in tqdm(issues):
    if i.number in data[repository][1]:
      idx = data[repository][1].index(i.number)
      data[repository][0][idx] = embed_query([template(i.title, i.body)], model)[0].embedding
    data[repository][0].append(
      embed_query([template(i.title, i.body)], model)[0].embedding
    )
    data[repository][1].append(i.number)
  for i in tqdm(pull_requests):
    if i.number in data[repository][1]:
      idx = data[repository][1].index(i.number)
      data[repository][0][idx] = embed_query([template(i.title, i.body)], model)[0].embedding
    data[repository][0].append(
      embed_query([template(i.title, i.body)], model)[0].embedding
    )
    data[repository][1].append(i.number)
  _ = Path("db.pkl").write_bytes(dumps(dict(data)))

if __name__ == "__main__":
  app()

