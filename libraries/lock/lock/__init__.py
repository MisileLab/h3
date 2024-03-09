from jwt import encode, decode

from typing import Callable, Optional
from datetime import datetime, timedelta

type opstr = Optional[str]

class JWTExpired(Exception):
  def __init__(self):
    super().__init__("JWT has expired")

class AuthError(Exception):
  def __init__(self):
    super().__init__("JWT username is different from provided username or password is incorrect")

class Instance:
  def __init__(self, login_with_password: Callable[[str, str], bool], secret_key: str, expire_time: timedelta = timedelta(minutes=30)):
    self.login_with_password = login_with_password
    self.secret_key = secret_key
    self.expire_time = expire_time

  def __create_jwt__(self, username: str) -> str:
    return encode({
      "username": username,
      "exp": (datetime.utcnow() + self.expire_time).timestamp()
    }, self.secret_key)
  
  def login(self, username: opstr, password: opstr, jwt: opstr) -> str:
    if jwt is None:
      if username is None or password is None:
        raise ValueError("username and password must be provided if jwt is not provided")
      if not self.login_with_password(username, password):
        raise AuthError()
      return self.__create_jwt__(username, self.secret_key)
    jwt = decode(jwt, self.secret_key)
    if jwt["exp"] < datetime.utcnow().timestamp():
      raise JWTExpired()
    if username is not None and username != jwt["username"]:
      raise AuthError()
    return jwt
