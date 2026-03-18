"""
Твоя модель для gap detection — встроить сюда.

Интерфейс:
  - init_gap_model() -> True если модель загружена
  - predict_gap(text: str) -> (position: int, total_words: int)
    position — индекс слова (0-based), где пропуск
    total_words — число слов в тексте
"""

_gap_model = None  # твоя модель


def init_gap_model() -> bool:
    """
    Загрузи свою модель сюда.
    Верни True если успешно, False — будет использоваться заглушка.
    """
    # === ПРИМЕР: PyTorch checkpoint ===
    # import torch
    # global _gap_model
    # _gap_model = torch.load("path/to/your/model.pt")
    # _gap_model.eval()
    # return True

    # === ПРИМЕР: HuggingFace ===
    # from transformers import AutoModelForTokenClassification, AutoTokenizer
    # global _gap_model
    # model_path = "your-username/your-gap-model"  # или локальный путь
    # _gap_model = {
    #     "model": AutoModelForTokenClassification.from_pretrained(model_path),
    #     "tokenizer": AutoTokenizer.from_pretrained(model_path),
    # }
    # return True

    return False  # пока модель не подключена — заглушка


def predict_gap(text: str) -> tuple[int, int] | None:
    """
    Предсказание позиции пропуска.
    Возвращает (position, total_words) или None (тогда main.py использует заглушку).
    """
    if _gap_model is None:
        return None

    words = text.strip().split()
    if not words:
        return (0, 0)

    # === ПРИМЕР: твоя логика инференса ===
    # Если модель возвращает индекс токена/слова с пропуском:
    # inputs = _gap_model["tokenizer"](text, return_tensors="pt")
    # with torch.no_grad():
    #     logits = _gap_model["model"](**inputs).logits
    # # например: position = logits.argmax(dim=-1).item() или по своей схеме
    # position = ...  # 0-based индекс слова
    # position = min(position, len(words) - 1)
    # return (position, len(words))

    return None


# main.py использует это имя
gap_model_predict = predict_gap
