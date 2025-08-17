from github import Github
from github.Auth import Token

from pathlib import Path

g = Github(auth=Token(Path("token.txt").read_text().strip()))

for i in g.get_user().get_repos():
  print(i, i.archived)
  if not i.archived:
    continue
  issues = i.get_issues(state="open")
  pr = i.get_pulls(state="open")
  print(issues.totalCount, pr.totalCount)
  if issues.totalCount == 0 and pr.totalCount == 0:
    continue
  i.edit(archived=False)
  for j in issues:
    print(f"closing issue {j.id}")
    j.edit(state="closed")
  for j in pr:
    print(f"closing pr {j.id}")
    j.edit(state="closed")
  i.edit(archived=True)
