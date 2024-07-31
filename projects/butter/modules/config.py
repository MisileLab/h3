from tomli import loads

from pathlib import Path

_config = loads(Path("./config.toml").read_text())
api_key = _config["token"]