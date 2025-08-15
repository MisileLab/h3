from github import Github
from github.Auth import Token

from pathlib import Path

g = Github(auth=Token(Path("token.txt").read_text().strip()))

for i in g.get_user().get_repos():
  print(i, i.archived)
  if not i.archived:
    continue
  issues = i.get_issues(state="open")
  if issues.totalCount == 0:
    continue
  pr = i.get_pulls(state="open")
  if i.archived and (issues.totalCount >= 0 or pr.totalCount >= 0):
    i.edit(archived=False)
    for j in issues:
      print(f"closing issue {j.id}")
      j.edit(state="closed")
    for j in pr:
      print(f"closing pr {j.id}")
      j.edit(state="closed")
    i.edit(archived=True)
