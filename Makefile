.PHONY: actions server chat cserver client speech

actions:
	uv run --env-file .env python -m rasa run actions

server:
	uv run python -m rasa run --enable-api --cors "*"

chat:
	uv run python -m rasa shell

speech client:
	uv run --env-file .env uvicorn server.speech_server:app --port 8001 --reload

cserver:
	pkill -f "rasa run actions" || true
	pkill -f "rasa run --enable-api" || true
