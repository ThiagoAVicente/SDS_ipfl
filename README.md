# UA Campus Assistant

Task-oriented spoken dialogue system for the University of Aveiro. Handles schedule queries, canteen lookup, canteen menus, exam information, and exam date saving.

Built with Rasa 3.6, Whisper ASR, and gTTS TTS. Requires Python 3.10.

## Setup

```bash
cp .env.example .env        # add GROQ_API_KEY if wanted
uv sync
uv run python -m rasa train
```

## Running

Three processes needed in separate terminals:

```bash
make actions   # custom actions server (port 5055)

# text-only (no browser)
make chat

# web ui
make server    # Rasa API server (port 5005)
make client    # speech server + serve client (port 8001) 
```

**Note:** `make client` serves the web UI client on `http://localhost:8001`.

## Optional: Groq LLM responses

Set in `.env`:
```
GROQ_API_KEY=gsk_...
ENABLE_GROQ_RESPONSE=true
```

Responses will be rephrased by `llama-3.1-8b-instant` while remaining constrained to the retrieved data.
