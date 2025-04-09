from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from jwt import encode, decode, exceptions as ex

from string import ascii_lowercase
from secrets import SystemRandom
from datetime import datetime

difficulty = 16
difficulty_hash = 24
percentage = 0.5

ph = PasswordHasher()
rand = SystemRandom()
jwt_key = rand.randbytes(256).hex()
alg = "HS512"
exptime = 60 * 60 * 24 * 30 # 30 days

def generate_key(n: int):
  return "".join(rand.choice(ascii_lowercase) for _ in range(n))

def generate_challenge(info: str):
  password = [generate_key(difficulty_hash) for _ in range(difficulty)]
  return encode({
    "payload": [
      ph.hash(generate_key(difficulty_hash) if rand.random() < percentage else i) for i in password
    ],
    "original": password,
    "info": info,
    "exp": int((datetime.now() - datetime(1970, 1, 1)).total_seconds()) + exptime
  }, key=jwt_key, algorithm=alg)

def verify_challenge(payload: str, answer: list[bool], info: str) -> bool:
  try:
    decoded: dict[str, str] = decode(payload, key=jwt_key, algorithms=[alg], options={
      "require": ["exp", "payload", "original", "info"],
      "verify_exp": True
    })
    _answer: list[bool] = []
    for k, v in zip(decoded["payload"], decoded["original"]):
      try:
        _ = ph.verify(k, v)
        _answer.append(True)
      except VerifyMismatchError:
        _answer.append(False)
    print(_answer, answer)
    return info == decoded["info"] and _answer == answer
  except (
    ex.InvalidSignatureError,
    ex.ExpiredSignatureError,
    ex.MissingRequiredClaimError
  ) as e:
    print(e)
    return False

