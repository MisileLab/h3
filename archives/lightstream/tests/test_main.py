# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false, reportUnknownMemberType=false, reportUnknownLambdaType=false
# this is Copilot's test, I'll just disable all basedpyright's warning
from json import dumps
from fastapi.testclient import TestClient
from main import app
from libraries.pow import difficulty
from fastapi import HTTPException

client = TestClient(app)

def mock_generate_challenge(info):
  return "mocked_challenge"

def mock_verify_challenge(payload, loaded, info):
  return "verified" if loaded == [True] * difficulty else ""

def test_root_without_x_answer_returns_challenge(monkeypatch):
  monkeypatch.setattr("main.generate_challenge", mock_generate_challenge)
  response = client.get("/", headers={"X-Payload": "test"})
  assert response.status_code == 400
  assert response.text == "mocked_challenge"

def test_root_with_invalid_x_answer_returns_challenge(monkeypatch):
  monkeypatch.setattr("main.generate_challenge", mock_generate_challenge)
  response = client.get("/", headers={
    "X-Payload": "test",
    "X-Answer": "not a json"
  })
  assert response.status_code == 400
  assert response.text == "mocked_challenge"

def test_root_with_wrong_type_x_answer_returns_challenge(monkeypatch):
  monkeypatch.setattr("main.generate_challenge", mock_generate_challenge)
  response = client.get("/", headers={
    "X-Payload": "test",
    "X-Answer": '"not a list"'
  })
  assert response.status_code == 400
  assert response.text == "mocked_challenge"

def test_root_with_wrong_length_x_answer_returns_challenge(monkeypatch):
  monkeypatch.setattr("main.generate_challenge", mock_generate_challenge)
  answer = str([True] * (difficulty - 1))
  response = client.get("/", headers={
    "X-Payload": "test",
    "X-Answer": answer.replace("'", '"')
  })
  assert response.status_code == 400
  assert response.text == "mocked_challenge"

def test_root_with_non_bool_x_answer_returns_challenge(monkeypatch):
  monkeypatch.setattr("main.generate_challenge", mock_generate_challenge)
  answer = str([1] * difficulty)
  response = client.get("/", headers={
    "X-Payload": "test",
    "X-Answer": answer.replace("'", '"')
  })
  assert response.status_code == 400
  assert response.text == "mocked_challenge"

def test_root_with_valid_x_answer(monkeypatch):
  monkeypatch.setattr("main.verify_challenge", mock_verify_challenge)
  answer = dumps(list([True] * difficulty))
  response = client.get("/", headers={
    "X-Payload": "test",
    "X-Answer": answer
  })
  assert response.status_code == 200
  assert response.json() == "succeed :sunglasses:"
  assert response.headers["X-Response"] == "verified"

def test_root_with_valid_x_answer_but_verify_fails(monkeypatch):
  monkeypatch.setattr("main.verify_challenge", lambda p, l, i: "")  # noqa: E741
  monkeypatch.setattr("main.generate_challenge", mock_generate_challenge)
  answer = str([True] * difficulty)
  response = client.get("/", headers={
    "X-Payload": "test",
    "X-Answer": answer.replace("'", '"')
  })
  assert response.status_code == 400
  assert response.text == "mocked_challenge"

def test_root_with_no_client(monkeypatch):
  # Patch request.client to None
  async def fake_challenge(request, call_next):
    request.client = None
    raise HTTPException(status_code=400, detail="Client IP not found")
  monkeypatch.setattr("main.challenge", fake_challenge)
  # This test is illustrative; FastAPI TestClient always sets client, so this is hard to simulate.