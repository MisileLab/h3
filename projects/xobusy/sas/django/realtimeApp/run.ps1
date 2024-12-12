python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
daphne -p 8000 realtimeApp.asgi:application
