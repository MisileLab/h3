from json import dumps
from pathlib import Path

from inquirer import prompt, Text # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

def main():
  p = Path("./config.json")
  if p.exists():
    print("Configuration file already exists.")
    return
  questions = [
    Text("openai_url", message="OpenAI API URL", default="https://api.openai.com/v1"),
    Text("openai_key", message="OpenAI API Key", default=""),
    Text("casual_model", message="Model for casual conversations", default="deepseek/deepseek-chat-v3-0324"),
    Text("high_reasoning_model", message="Model for high reasoning tasks", default="deepseek/deepseek-r1-0528")
  ]
  answers: dict[str, str] = prompt(questions) # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
  if None in answers.values():
    print("Configuration not completed.")
    return
  _ = p.write_text(dumps(answers))

if __name__ == "__main__":
  main()

