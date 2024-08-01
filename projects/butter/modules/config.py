from tomli import loads

from pathlib import Path

config = loads(Path("./config.toml").read_text())
api_key = config["token"]
