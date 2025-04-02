from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from jwt import encode, decode, exceptions as ex

from string import ascii_lowercase
from secrets import SystemRandom
from datetime import datetime

difficulty = 1

ph = PasswordHasher()
rand = SystemRandom()
length = len(ascii_lowercase)
jwt_key = rand.randbytes(64).hex()
alg = "HS512"
exptime = 60 * 60 * 24 * 30 # 30 days

def generate_challenge():
  password = "".join([ascii_lowercase[
    rand.randint(0, length - 1)
  ] for _ in range(difficulty)])
  print(password)
  return encode({
    "payload": ph.hash(password),
    "exp": (datetime.now() - datetime(1970, 1, 1)).total_seconds() + exptime
  }, key=jwt_key, algorithm=alg)

def verify_challenge(payload: str, answer: str) -> bool:
  try:
    decoded: dict[str, str] = decode(payload, key=jwt_key, algorithms=[alg], options={
      "require": ["exp", "payload"],
      "verify_exp": True
    })
    try:
      _ = ph.verify(decoded["payload"], answer)
    except VerifyMismatchError:
      return False
    return True
  except (
    ex.InvalidSignatureError,
    ex.ExpiredSignatureError,
    ex.MissingRequiredClaimError
  ) as e:
    print(e)
    return False

