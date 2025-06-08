from polars import read_avro, col, DataFrame, concat
from gradio import Blocks, Row, Textbox, Button, Markdown, Column, Image

from utils import Data, ProcessedData, read_cached_avro

comments = read_avro("comments.avro")
df = read_cached_avro("processed.avro")

start_idx = 0

def append(df: DataFrame, data: ProcessedData) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

while len(df) != 0 and not df.filter(col("comment_id") == comments[start_idx]["comment_id"][0]).is_empty(): # pyright: ignore[reportAny]
  start_idx += 1

print(start_idx)

def comment_to_return(data: Data):
  existing_comments = comments.filter(col("comment_id") == data.parent_id)
  parent_comment: Data
  if existing_comments.is_empty():
    parent_comment = Data(
      comment_id="",
      content="",
      author_name="",
      author_image_url="",
      video_id=""
    )
  else:
    parent_comment = Data.model_validate(existing_comments.to_dicts()[0])

  # Ensure image URLs are valid or return None
  parent_image = parent_comment.author_image_url if parent_comment.author_image_url else None
  current_image = data.author_image_url if data.author_image_url else None

  return [
    data.video_id,
    parent_comment.author_name,
    parent_image,
    parent_comment.content,
    data.author_name,
    current_image,
    data.content
  ]

def classify_comments(is_bot: bool):
  global df, start_idx
  data = Data.model_validate(comments[start_idx].to_dicts()[0])
  processed_data = ProcessedData(
    **data.model_dump(), # pyright: ignore[reportAny]
    is_bot_comment=is_bot
  )
  df = append(df, processed_data)
  start_idx += 1
  df.write_avro("processed.avro")
  progress = f"Comment {start_idx + 1} of {len(comments)}"
  next_data = Data.model_validate(comments[start_idx].to_dicts()[0])
  return [*comment_to_return(next_data), progress]

def get_data():
  data = Data.model_validate(comments[start_idx].to_dicts()[0])
  return [*comment_to_return(data), f"Comment {start_idx + 1} of {len(comments)}"]

def classify_as_bot():
  return classify_comments(True)

def classify_as_human():
  return classify_comments(False)

with Blocks(title="Comment Classification Tool", theme="soft") as frontend:
  _ = Markdown("# Comment Classification Tool")
  _ = Markdown("Classify comments as **Bot** or **Human** based on the content and context.")

  with Row():
    # Progress indicator
    progress_text = Textbox(
      label="Progress",
      value=f"Comment {start_idx + 1} of {len(comments)}",
      interactive=False
    )

  with Row():
    # Parent comment section
    with Column(scale=1):
      _ = Markdown("## Parent Comment")
      parent_author = Textbox(label="Parent Author", interactive=False)
      parent_image = Image(show_label=False, height=60, width=60)
      parent_content = Textbox(
        label="Parent Content",
        lines=4,
        interactive=False,
        placeholder="No parent comment"
      )

    # Current comment section
    with Column(scale=1):
      _ = Markdown("## Current Comment (Classify This)")
      current_author = Textbox(label="Current Author", interactive=False)
      current_image = Image(show_label=False, height=60, width=60)
      current_content = Textbox(
        label="Current Content",
        lines=4,
        interactive=False
      )
      video_id = Textbox(label="Video ID", interactive=False)

  with Row():
    # Classification buttons
    bot_button = Button("🤖 Bot Comment", variant="secondary", size="lg")
    human_button = Button("👤 Human Comment", variant="primary", size="lg")

  # Output components for updating the display
  outputs = [
    video_id,
    parent_author,
    parent_image,
    parent_content,
    current_author,
    current_image,
    current_content,
    progress_text
  ]

  # Button event handlers
  _ = bot_button.click(
    fn=classify_as_bot,
    outputs=outputs
  )

  _ = human_button.click(
    fn=classify_as_human,
    outputs=outputs
  )

  # Load initial data when the interface starts
  _ = frontend.load(
    fn=get_data,
    outputs=outputs
  )

if __name__ == "__main__":
  _ = frontend.launch()

