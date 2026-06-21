import io
import os
import re
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from groq import Groq

_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
_groq = Groq(api_key=_GROQ_KEY) if _GROQ_KEY else None

_PROMPT = "Assistente universitário da Universidade de Aveiro. Horários, cantinas, exames, DETI, ESTGA, Crasto, Santiago."


def _preprocess_tts(text: str) -> str:
    # Room numbers like 04.2.25 → 04-2-25 (prevents gTTS reading as date)
    text = re.sub(r'\b(\d{1,3})\.(\d{1,2})\.(\d{2,3})\b', r'\1-\2-\3', text)
    return text


@app.post("/tts")
async def text_to_speech(body: dict):
    text = _preprocess_tts(body.get("text", ""))
    lang = body.get("lang", "pt")
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


@app.post("/asr")
async def speech_to_text(file: UploadFile = File(...)):
    if not _groq:
        return JSONResponse({"error": "GROQ_API_KEY not set"}, status_code=503)

    suffix = ".webm"
    if file.filename:
        _, ext = os.path.splitext(file.filename)
        if ext:
            suffix = ext

    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = _groq.audio.transcriptions.create(
                file=(os.path.basename(tmp_path), f),
                model="whisper-large-v3-turbo",
                language="pt",
                prompt=_PROMPT,
                response_format="json",
            )
        return JSONResponse({"text": result.text.strip()})
    finally:
        os.unlink(tmp_path)


# Serve client/ as static files — must be last (catches all remaining routes)
_client_dir = os.path.join(os.path.dirname(__file__), "..", "client")
app.mount("/", StaticFiles(directory=_client_dir, html=True), name="static")
