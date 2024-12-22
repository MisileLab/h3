from satellite_py import DB
from tomli import loads
from boto3 import client # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pathlib import Path
from typing import final

@final
class Initalizer:
  def __init__(self):
    self.c = DB()
    self.config = loads(Path("./config.toml").read_text())
    self.smtp_client = client( # pyright: ignore[reportUnknownMemberType]
      'ses',
      region_name="eu-west-2",
      aws_access_key_id=self.config["aws"]["key"],
      aws_secret_access_key=self.config["aws"]["secret_key"]
    )
    self.key = self.config["security"]["key"]

initializer = Initalizer()

