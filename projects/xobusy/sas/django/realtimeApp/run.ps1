python -m venv .venv
pip install -r requirements.txt
./.venv/bin/daphne -p 8000 realtimeApp.asgi:application
