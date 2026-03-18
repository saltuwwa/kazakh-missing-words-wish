# Тілқат — тәуелділіктер

Қосымшаны іске қосу үшін мына пакеттер қажет.

## Кеңейтілген тізім (requirements.txt үшін)

```
fastapi
uvicorn[standard]
openai-whisper
transformers
torch
python-multipart
```

## Орнату

```bash
pip install -r requirements.txt
```

## Іске қосу

```bash
cd web
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Содан кейін браузерде: http://localhost:8000
