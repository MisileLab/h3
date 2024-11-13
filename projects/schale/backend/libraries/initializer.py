from edgedb import create_async_client
from tomli import loads
from boto3 import client

from pathlib import Path

class Initalizer:
  def __init__(self):
    self.c = create_async_client()
    self.config = loads(Path("./config.toml").read_text())
    self.smtp_client = client(
      'ses',
      region_name="eu-west-2",
      aws_access_key_id=self.config["aws"]["key"],
      aws_secret_access_key=self.config["aws"]["secret_key"]
    )
    self.key = self.config["security"]["key"]

initializer = Initalizer()

