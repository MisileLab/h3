# pyright: reportUnusedParameter=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# this is Copilot's test, I'll just disable all basedpyright's warning
from string import ascii_lowercase

from argon2.exceptions import VerifyMismatchError
import jwt

import libraries.pow as pow_mod

def test_generate_key_length_and_charset():
  key = pow_mod.generate_key(10)
  assert len(key) == 10
  assert all(c in ascii_lowercase for c in key)

def test_generate_challenge_and_decode_jwt():
  info = "test_info"
  challenge = pow_mod.generate_challenge(info)
  decoded = pow_mod.decode_jwt(challenge)
  assert "payload" in decoded
  assert "original" in decoded
  assert "info" in decoded
  assert "exp" in decoded
  assert decoded["info"] == info
  assert isinstance(decoded["payload"], list)
  assert isinstance(decoded["original"], list)
  assert len(decoded["payload"]) == pow_mod.difficulty
  assert len(decoded["original"]) == pow_mod.difficulty

def test_verify_challenge_success_and_confirmed():
  info = "verify_test"
  challenge = pow_mod.generate_challenge(info)
  decoded = pow_mod.decode_jwt(challenge)
  # Simulate correct answer: True if hash matches original, else False
  answer = []
  for k, v in zip(decoded["payload"], decoded["original"]):
    try:
      _ = pow_mod.ph.verify(k, v)
      answer.append(True)
    except VerifyMismatchError:
      answer.append(False)
  # First verification (should add confirmed)
  verified = pow_mod.verify_challenge(challenge, answer, info)
  assert verified
  decoded_verified = pow_mod.decode_jwt(verified)
  assert decoded_verified.get("confirmed") is True
  # Second verification (already confirmed, should return same payload)
  verified2 = pow_mod.verify_challenge(verified, answer, info)
  assert verified2 == verified

def test_verify_challenge_wrong_info_returns_empty():
  info = "info1"
  challenge = pow_mod.generate_challenge(info)
  decoded = pow_mod.decode_jwt(challenge)
  answer = []
  for k, v in zip(decoded["payload"], decoded["original"]):
    try:
      _ = pow_mod.ph.verify(k, v)
      answer.append(True)
    except VerifyMismatchError:
      answer.append(False)
  result = pow_mod.verify_challenge(challenge, answer, "wrong_info")
  assert result == ""

def test_verify_challenge_none_answer_returns_empty():
  info = "info2"
  challenge = pow_mod.generate_challenge(info)
  result = pow_mod.verify_challenge(challenge, None, info)
  assert result == ""

def test_decode_jwt_invalid_signature():
  # Tamper with a valid token
  info = "tamper"
  challenge = pow_mod.generate_challenge(info)
  tampered = challenge[:-1] + ("a" if challenge[-1] != "a" else "b")
  result = pow_mod.verify_challenge(tampered, [], info)
  assert result == ""

def test_decode_jwt_missing_claims():
  # Create a JWT missing required claims
  payload = {"foo": "bar"}
  token = jwt.encode(payload, key=pow_mod.jwt_key, algorithm=pow_mod.alg)
  result = pow_mod.verify_challenge(token, [], "info")
  assert result == ""