from pydantic import BaseModel
from pyyoutube import Api, Comment, CommentThread # pyright: ignore[reportMissingTypeStubs]
from polars import DataFrame, read_avro, concat, col, String
from os import getenv
from pathlib import Path

client = Api(api_key=getenv("YOUTUBE_API_KEY"))
videos = read_avro("videos.avro")
df = read_avro("comments.avro") if Path("comments.avro").exists() else DataFrame()

class Data(BaseModel):
  comment_id: str
  content: str
  author_name: str
  author_image_url: str
  video_id: str
  parent_id: str | None = None

schema = {
  "comment_id": String,
  "content": String,
  "author_name": String,
  "author_image_url": String,
  "video_id": String,
  "parent_id": String
}

def append(df: DataFrame, data: Data) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

def append_comment(df: DataFrame, comment: Comment, video_id: str) -> DataFrame:
  snippet = comment.snippet
  if snippet is None:
    raise ValueError("Comment has no snippet")
  comment_id = comment.id
  if comment_id is None:
    raise ValueError("Comment has no ID")
  content = snippet.textDisplay
  if content is None:
    raise ValueError("Comment has no content")
  authorDisplayName = snippet.authorDisplayName
  if authorDisplayName is None:
    raise ValueError("Comment has no author display name")
  authorImageUrl = snippet.authorProfileImageUrl
  if authorImageUrl is None:
    raise ValueError("Comment has no author image URL")
  return append(df, Data(
    comment_id=comment_id,
    content=content,
    author_name=authorDisplayName,
    author_image_url=authorImageUrl,
    parent_id=snippet.parentId,
    video_id=video_id
  ))

def append_commentThreads(df: DataFrame, commentThread: CommentThread, video_id: str) -> DataFrame:
  snippet = commentThread.snippet
  if snippet is None:
    raise ValueError("CommentThread has no snippet")
  topLevelComment = snippet.topLevelComment
  if topLevelComment is None:
    raise ValueError("CommentThread has no top-level comment")
  topLevelComment_id = topLevelComment.id
  if topLevelComment_id is None:
    raise ValueError("Top-level comment has no ID")
  df = append_comment(df, topLevelComment, video_id)
  _replies = commentThread.replies
  replies = _replies.comments if _replies is not None else []
  if not replies:
    replies = []
  if len(replies) == snippet.totalReplyCount:
    for reply in replies:
      df = append_comment(df, reply, video_id)
  else:
    comments = client.get_comments( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
      parts="id,snippet",
      parent_id=topLevelComment_id,
    )
    if isinstance(comments, dict):
      raise ValueError("Failed to fetch comments for parent ID: " + topLevelComment_id)
    items = comments.items
    if items is None:
      items = []
    for reply in items:
      df = append_comment(df, reply, video_id)
  return df

for i in videos.iter_rows(named=True):
  video_id: str = i["videoId"] # pyright: ignore[reportAny]
  if len(df) != 0 and df.filter(col("videoId") == video_id).height > 0: # pyright: ignore[reportUnknownMemberType]
    continue
  print(video_id)
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
    continue

  for i in items:
    df = append_commentThreads(df, i, video_id)
