from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from libraries.pow import generate_challenge, verify_challenge, difficulty

from jwt import decode

from time import perf_counter
from string import ascii_lowercase

challenge = generate_challenge()
payload: str = decode(challenge, options={"verify_signature": False})["payload"]
ph = PasswordHasher()
answer = ""

def solve(value: str) -> bool:
  if len(value) != difficulty:
    for i in ascii_lowercase:
      print(value + i)
      if solve(value + i):
        return True
    return False
  for i in ascii_lowercase:
    try:
      _ = ph.verify(payload, value + i)
    except VerifyMismatchError:
      continue
    else:
      print("solved")
      global answer
      answer = value + i
      return True
  return False

start = perf_counter()
_ = solve("")
print(verify_challenge(challenge, answer))
print(answer, perf_counter() - start)

