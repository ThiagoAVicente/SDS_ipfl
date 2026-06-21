.PHONY: actions server chat cserver

actions:
	uv run python -m rasa run actions

server:
	uv run python -m rasa run --enable-api

chat:
	uv run python -m rasa shell

cserver:
	pkill -f "rasa run actions" || true
	pkill -f "rasa run --enable-api" || true
