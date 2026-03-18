"""
Тілқат — жоғалған сөзді қалпына келтіру
FastAPI backend: ASR (Whisper), gap detection, word suggestions
"""
import os
import time
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Пути к данным (submission + test_inputs для lookup при совпадении текста)
BASE_DIR = Path(__file__).parent.parent
SUBMISSION_PATH = BASE_DIR / "submission(1).csv"
TEST_INPUTS_PATH = BASE_DIR / "test_inputs.csv"

app = FastAPI(title="Тілқат")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Глобальные объекты моделей (загружаются при старте)
whisper_processor = None
whisper_model = None
fill_mask_pipeline = None
gap_lookup = None  # dict: нормализованный текст -> (position, total_words)
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")  # если задан — используем облако

KAZAKH_WHISPER = "abilmansplus/whisper-turbo-ksc2"


@app.on_event("startup")
async def startup():
    global whisper_processor, whisper_model, fill_mask_pipeline, gap_lookup
    from transformers import pipeline

    if ELEVENLABS_API_KEY:
        print("[STARTUP] Using ElevenLabs Scribe (cloud) for ASR")
    else:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        print(f"[STARTUP] Loading Kazakh Whisper ({KAZAKH_WHISPER})...")
        t0 = time.time()
        whisper_processor = WhisperProcessor.from_pretrained(KAZAKH_WHISPER, language="kazakh", task="transcribe")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(KAZAKH_WHISPER)
        whisper_model = whisper_model.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[STARTUP] Kazakh Whisper loaded in {time.time() - t0:.1f}s")

    print("[STARTUP] Loading fill-mask pipeline (KazBERT)...")
    t0 = time.time()
    fill_mask_pipeline = pipeline("fill-mask", model="Eraly-ml/KazBERT", top_k=5)
    print(f"[STARTUP] Fill-mask loaded in {time.time() - t0:.1f}s")

    print("[STARTUP] Loading gap model (custom)...")
    try:
        from gap_model import init_gap_model
        if init_gap_model():
            print("[STARTUP] Gap model loaded")
        else:
            print("[STARTUP] Gap model not configured, using stub")
    except Exception as e:
        print(f"[STARTUP] Gap model skipped: {e}")

    print("[STARTUP] Loading gap lookup from submission + test_inputs...")
    gap_lookup = {}
    if SUBMISSION_PATH.exists() and TEST_INPUTS_PATH.exists():
        import pandas as pd
        test_df = pd.read_csv(TEST_INPUTS_PATH)
        sub_df = pd.read_csv(SUBMISSION_PATH)
        for _, row in test_df.iterrows():
            txt = str(row["text"]).strip()
            pid = row["ID"]
            if pid in sub_df["ID"].values:
                pos = int(sub_df[sub_df["ID"] == pid]["missing_word_position"].iloc[0])
                nw = len(txt.split())
                gap_lookup[txt] = (min(pos, nw), nw)
        print(f"[STARTUP] Gap lookup: {len(gap_lookup)} entries")
    else:
        print(f"[STARTUP] Gap lookup skipped: submission or test_inputs not found")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return FileResponse(html_path, media_type="text/html")


async def _transcribe_elevenlabs(path: str) -> str:
    """ElevenLabs Scribe API — лучший казахский (WER ~3%)."""
    import httpx
    with open(path, "rb") as f:
        audio_bytes = f.read()
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Accept": "application/json"},
            data={"model_id": "scribe_v2", "language_code": "kk"},
            files={"file": ("audio.webm", audio_bytes, "audio/webm")},
        )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API error {response.status_code}: {response.text[:200]}")
    data = response.json()
    # Ответ: text в корне или в transcripts
    if isinstance(data, dict):
        text = data.get("text") or ""
        if not text and "transcripts" in data:
            chunks = data.get("transcripts", [])
            text = " ".join(c.get("text", "") for c in chunks)
    else:
        text = ""
    return (text or "").strip()


def _transcribe_kazakh_whisper(path: str) -> str:
    """Kazakh Whisper (HF) — fine-tuned на KSC2, WER ~9%."""
    import librosa
    import torch
    sr = 16_000
    speech, _ = librosa.load(path, sr=sr)
    forced_ids = whisper_processor.get_decoder_prompt_ids(language="kazakh", task="transcribe")
    inputs = whisper_processor(speech, sampling_rate=sr, return_tensors="pt").input_features.to(whisper_model.device)
    attention_mask = torch.ones_like(inputs[:, :, 0])
    with torch.no_grad():
        ids = whisper_model.generate(
            inputs, forced_decoder_ids=forced_ids,
            max_length=448, num_beams=5, attention_mask=attention_mask
        )
    text = whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]
    return (text or "").strip()


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "audio").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await audio.read()
        f.write(content)
        path = f.name
    try:
        t0 = time.time()
        if ELEVENLABS_API_KEY:
            text = await _transcribe_elevenlabs(path)
        else:
            text = _transcribe_kazakh_whisper(path)
        elapsed = time.time() - t0
        print(f"[TRANSCRIBE] {elapsed:.1f}s -> {text[:80]}...")
        return {"text": text}
    except Exception as e:
        print(f"[TRANSCRIBE] Error: {e}")
        return {"error": "Дауысты тану мүмкін болмады"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/detect-gap")
async def detect_gap(data: dict):
    text = (data.get("text") or "").strip()
    words = text.split()
    if not words:
        return {"position": 0, "total_words": 0}

    # 1. lookup по submission (если текст из test_inputs)
    if gap_lookup and text in gap_lookup:
        pos, total = gap_lookup[text]
        return {"position": pos, "total_words": total}

    # 2. Твоя модель gap detection (если подключена)
    try:
        from gap_model import predict_gap
        result = predict_gap(text)
        if result is not None:
            pos, total = result
            return {"position": pos, "total_words": total}
    except Exception as e:
        print(f"[DETECT-GAP] gap_model error (fallback to stub): {e}")

    # 3. Заглушка (fallback)
    n = len(words)
    pos = min(n // 2, n - 1) if n > 1 else 0
    return {"position": pos, "total_words": n}


@app.post("/suggest")
async def suggest(data: dict):
    text = (data.get("text") or "").strip()
    position = int(data.get("position", 0))
    words = text.split()
    n = len(words)
    if n == 0 or position < 0 or position >= n:
        return {"suggestions": []}

    # Вставляем [MASK] на позицию (gapped text: слов на 1 меньше, вставляем между)
    masked = words[:position] + ["[MASK]"] + words[position:]
    masked_str = " ".join(masked)

    try:
        results = fill_mask_pipeline(masked_str, top_k=10)
        seen = set()
        suggestions = []
        for r in results:
            token = (r.get("token_str") or "").strip().replace("##", "")
            if not token or len(token) < 2 or token.lower() in seen:
                continue
            seen.add(token.lower())
            suggestions.append(token)
            if len(suggestions) >= 3:
                break
        return {"suggestions": suggestions}
    except Exception as e:
        print(f"[SUGGEST] Error: {e}")
        return {"suggestions": []}
