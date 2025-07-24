insert Data {
  author_name := <str>$author_name,
  content := <str>$content,
  is_bot := <bool>$is_bot
} unless conflict on (.author_name, .content) else (
  update Data filter .author_name = <str>$author_name and .content = <str>$content set {
    is_bot := <bool>$is_bot
  }
)
