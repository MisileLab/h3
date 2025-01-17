#!/usr/bin/env nu
loop {
  let i = input "id: ";
  print $i;
  infisical run -- uv run twscrape user_tweets --limit 1 $i | jq '.rawContent, .hashtags, .links, .media' -C | less -r
  sleep 1sec
}
