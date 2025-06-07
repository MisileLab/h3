from pyyoutube import Api # pyright: ignore[reportMissingTypeStubs]
from polars import DataFrame, read_avro, concat
from os import getenv

client = Api(api_key=getenv("YOUTUBE_API_KEY"))
videos = read_avro("videos.avro")
df = DataFrame()

def append(df: DataFrame, data: dict[str, object]) -> DataFrame:
  return concat([df, DataFrame(data)], how="vertical", rechunk=True)

# TODO: skip existing datas
for i in videos.iter_rows(named=True):
  video_id: str = i["videoId"] # pyright: ignore[reportAny]
  comments = client.get_comment_threads( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    video_id=video_id,
    count=None,
    parts="replies,snippet,id",
    order="relevance"
  )
  if isinstance(comments, dict):
    exit(1)

  items = comments.items
  if not items:
    print("No videos found.")
    exit(1)

  for i in items:
    snippet = i.snippet
    replies = i.replies
    if replies is None or snippet is None:
      continue
    topLevelComment = snippet.topLevelComment
    if topLevelComment is None:
      continue
    topLevelComment_id = topLevelComment.id
    textDisplay = ""
    if topLevelComment_id:
      print("found top level comment")
      topLevelComment = client.get_comment_by_id( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        parts="snippet",
        comment_id=topLevelComment_id
      )
      if not isinstance(topLevelComment, dict):
        items = topLevelComment.items
        if not items:
          items = []
        topLevelComment_snippet = items[0].snippet
        if topLevelComment_snippet is None:
          textDisplay = ""
        else:
          textDisplay = topLevelComment_snippet.textDisplay
          if textDisplay is None:
            textDisplay = ""
    comments = replies.comments if replies.comments else []
    df = append(df, {
      "videoId": video_id,
      "totalReplyCount": snippet.totalReplyCount,
      "content": textDisplay,
      "replies": replies.comments
    })
    df.write_avro("comments.avro")
    if topLevelComment_id == snippet.totalReplyCount != len(comments):
      print("get all comments of video")
      comments = []
      # TODO: get all comments

