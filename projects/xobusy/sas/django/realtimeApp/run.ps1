python -m venv .venv
.\.venv\bin\pip install -r requirements.txt
.\.venv\bin\daphne -p 8000 realtimeApp.asgi:application
