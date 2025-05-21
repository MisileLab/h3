from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from libraries.pow import generate_challenge, verify_challenge, difficulty

from jwt import decode

from time import perf_counter
from threading import Thread

challenge = generate_challenge("")
jwt_decoded: dict[str, list[str]] = decode(challenge, options={"verify_signature": False}) # pyright: ignore[reportAny]
print(jwt_decoded)
original: list[str] = jwt_decoded["original"]
payload: list[str] = jwt_decoded["payload"]
ph = PasswordHasher()
result: dict[int, bool] = {}

def solve(index: int):
  try:
    _ = ph.verify(payload[index], original[index])
    result[index] = True
  except VerifyMismatchError:
    result[index] = False

start = perf_counter()
threads: list[Thread] = []
for i in range(difficulty):
  thread = Thread(target=solve, args=(i,))
  thread.start()
  threads.append(thread)
for thread in threads:
  thread.join()
print(verify_challenge(challenge, [result[i] for i in range(len(original))], ""))
print(perf_counter() - start)

