# Тілқат — жоғалған сөзді қалпына келтіру

Қазақша дауыс/аудио → транскрибация → пропуск табу → сөз нұсқалары.

## Іске қосу

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Қолданба: http://localhost:8000

## Құрылым

- `main.py` — FastAPI backend (Whisper, gap detection, fill-mask)
- `index.html` — UI (қазақ тілінде)
- `submission(1).csv` + `test_inputs.csv` — lookup для тестовых текстов (из корня проекта)

## Реальная модель для gap detection

Сейчас используется заглушка + lookup по `submission(1).csv` при совпадении с `test_inputs`. Для произвольного текста — stub.

Для production: загрузить обученную модель из `output/model_seed42/` и делать инференс как в `kazakh_missing_words_vastai.ipynb`.
