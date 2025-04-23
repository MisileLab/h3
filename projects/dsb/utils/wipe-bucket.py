from pathlib import Path

from minio import Minio
from minio.deleteobjects import DeleteObject
from tomli import loads

config: dict[str, str] = loads(Path("config.toml").read_text())["minio"] # pyright: ignore[reportAny]
client = Minio(
  "localhost:9000",
  access_key = config["name"],
  secret_key = config["password"],
  secure = False
)

print("starting")
result = client.remove_objects("attic", map(
  lambda x: DeleteObject(x.object_name if x.object_name is not None else "none-exist"),
  client.list_objects("attic"))
)
for error in result:
  print(error)
