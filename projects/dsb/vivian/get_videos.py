from pyyoutube import Api # pyright: ignore[reportMissingTypeStubs]
from polars import DataFrame
from os import getenv

client = Api(api_key=getenv("YOUTUBE_API_KEY"))
videos = client.get_videos_by_chart( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
  chart="mostPopular",
  region_code="KR",
  count=None
)

if isinstance(videos, dict):
  exit(1)

items = videos.items
if not items:
  print("No videos found.")
  exit(1)

df = DataFrame({
  "videoId": [item.id for item in items]
})
df.write_avro("videos.avro")

