# 🏆 WISH Hackathon 2026 — Полный разбор кейсов

> **Дедлайн сабмита:** 15 марта, 12:00  
> **Питчинг Top-20:** 15 марта, ~13:30 (до 7 минут)  
> **Платформа:** Kaggle

---

## ⚠️ Критически важные правила (оба кейса)

| Правило | Что значит на практике |
|---------|----------------------|
| ✅ Внешние датасеты разрешены | Качай любые казахские корпуса с HuggingFace, GitHub, CommonCrawl |
| ✅ Pretrained модели разрешены | xlm-roberta, KazBERT, mT5 — всё можно |
| ✅ LLM как инструмент (Claude, ChatGPT) | Можно для кода, идей, понимания грамматики |
| ❌ LLM API для предсказаний — **ЗАПРЕЩЕНО** | Нельзя прогнать тест через OpenAI/Anthropic API и сабмитить |
| ❌ Ручная разметка теста — запрещена | Модель должна обобщать, не угадывать вручную |
| 🔁 Inference воспроизводим в Kaggle Notebook | Код должен работать без ошибок и давать те же предсказания |
| 👥 До 4 человек в команде | Все должны быть зарегистрированы на Kaggle |
| 📤 До 40 сабмитов в день | Итерируй часто, не бойся сабмитить |

> 💡 **Главный вывод из правил:** нельзя использовать GPT-4/Claude API для генерации предсказаний. Но можно использовать **открытые модели** (llama, mistral, qwen) локально — в том числе большие. Это открывает возможности.

---

## Case 1 — Kazakh Punctuation Restoration

🔗 [kaggle.com/competitions/kaz-punct-hackathon](https://www.kaggle.com/competitions/kaz-punct-hackathon/overview)

### 📌 Задача

Token classification: для каждого слова предсказать знак препинания **после него**.

| Лейбл | Значение |
|-------|----------|
| `O` | Ничего (~85% токенов) |
| `COMMA` | Запятая `,` |
| `PERIOD` | Точка `.` или `!` |
| `QUESTION` | Вопрос `?` |

```
Input:  сәлем менің атым бақыт қалың қалай
Output: COMMA   O     O   PERIOD  O   QUESTION
→ "Сәлем, менің атым Бахыт. Қалың қалай?"
```

### 📂 Данные

| Файл | Строк | Описание |
|------|-------|----------|
| `train_example.csv` | **500** | Только формат — не трейн! |
| `test.csv` | **3 552** | Тест без лейблов |
| `sample_submission.csv` | 3 552 | Шаблон (всё `O` = 0.000) |

> ⚠️ **Трейн строишь сам** из казахских текстов с пунктуацией.

### 🗂 Как построить трейн

**Лучший источник:** `kz-transformers/multidomain-kazakh-dataset` на HuggingFace  
(Wikipedia + новости + книги + веб — разные домены, как в тесте)

```python
def strip_and_label(text):
    tokens = text.split()
    labels, clean_tokens = [], []
    for token in tokens:
        if token.endswith('?'):
            labels.append('QUESTION')
            clean_tokens.append(token[:-1].lower())
        elif token.endswith(('.', '!')):
            labels.append('PERIOD')
            clean_tokens.append(token[:-1].lower())
        elif token.endswith(','):
            labels.append('COMMA')
            clean_tokens.append(token[:-1].lower())
        else:
            labels.append('O')
            clean_tokens.append(token.lower())
    return ' '.join(clean_tokens), ' '.join(labels)
```

### 📊 Метрика: Macro F1

- Считается только по `{COMMA, PERIOD, QUESTION}` — `O` **исключён**
- Все три класса весят **одинаково**
- Предсказание только `O` → скор **0.000**
- `COMMA` — самый сложный (неоднозначен даже для людей)

### 🛠 Технический план

```
Час 0–1   → Строим трейн из multidomain-kazakh-dataset (strip_and_label)
Час 1–2   → Baseline xlm-roberta-base, первый сабмит (любой скор > 0)
Час 2–4   → Fine-tune KazBERT, class weights для COMMA и QUESTION
Час 4–6   → Domain augmentation, ансамбль моделей
Час 6–7   → Финальный сабмит + начало презентации
```

**Важно перед каждым сабмитом:**
```python
# Кол-во лейблов должно точно совпадать с кол-вом токенов
for _, row in test.iterrows():
    expected = len(row['input_text'].split())
    actual   = len(sub.loc[sub['id'] == row['id'], 'labels'].values[0].split())
    assert expected == actual, f"Mismatch on {row['id']}"
```

### ✅ Плюсы
- Метрика полностью известна → чёткая цель оптимизации
- Богатая экосистема HuggingFace для token classification
- Сильный питчинг: "восстановление пунктуации для казахских ASR-систем"
- Высокий потолок: ensemble + CRF + domain diversity

### ❌ Риски
- 1–2 часа только на построение трейна (потеря времени на хакатоне)
- `COMMA` сложен даже для хороших моделей
- Тест многодоменный → нужна diversity в трейне, иначе просадка

---

## Case 2 — Kazakh Missing Words Challenge

🔗 [kaggle.com/t/a0b8ceabd14c40e2b8973aa74dfd51c2](https://www.kaggle.com/t/a0b8ceabd14c40e2b8973aa74dfd51c2)

### 📌 Задача

Предсказать **позицию** (индекс, 0-based) удалённого слова в предложении.

```
Оригинал:  Мен дүкенге барып келдім.   → слова: [Мен, дүкенге, барып, келдім]
После:     Мен барып келдім.            → убрали слово на позиции 1
Предсказать: 1
```

> Задача — понять **синтаксис и порядок слов**, не угадать само слово.

### 📂 Данные

| Файл | Строк | Описание |
|------|-------|----------|
| `train.csv` | **550 000+** | Полные предложения — убираешь слово сам |
| `test_inputs.csv` | **15 000** | Предложения с пропуском, предсказать позицию |
| `sample_submission.csv` | 15 000 | Шаблон |

### 🗂 Как построить трейн из `train.csv`

```python
import pandas as pd, random

df = pd.read_csv('train.csv')
samples = []

for text in df['full_text']:
    words = text.split()
    if len(words) < 3:
        continue
    pos = random.randint(0, len(words) - 1)
    new_words = words[:pos] + words[pos+1:]
    samples.append({
        'text': ' '.join(new_words),
        'missing_word_position': pos
    })

train_df = pd.DataFrame(samples)
```

### 📊 Метрика: Accuracy

- Верно только при **точном совпадении** позиции
- 15 000 тестовых примеров
- Простая и прозрачная метрика

### 🛠 Технический план

```
Час 0       → EDA: длины предложений, распределение позиций
Час 0–0.5   → Baseline (предсказываем середину), первый сабмит
Час 0.5–2   → Fine-tune xlm-roberta-base как классификатор позиций
Час 2–4     → KazBERT, span scoring, augmentation (несколько позиций per sentence)
Час 4–6     → Ансамбль, улучшение accuracy
Час 6–7     → Финальный сабмит + презентация
```

```python
# Baseline (< 30 мин)
test = pd.read_csv('test_inputs.csv')
test['missing_word_position'] = test['text'].apply(lambda x: len(x.split()) // 2)
test[['ID', 'missing_word_position']].to_csv('submission.csv', index=False)

# Основная модель
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base", num_labels=50  # макс. позиция
)
```

### ✅ Плюсы
- **550к строк трейна уже готово** — не тратишь время на сборку
- Baseline за **< 30 минут**
- Метрика Accuracy — прозрачная и понятная
- Меньше конкурентов → реальнее Top-20
- Отличный питчинг: "понимание синтаксиса казахского языка"

### ❌ Риски
- Accuracy строгая: позиция ±1 не засчитывается
- Гибкий порядок слов в казахском → несколько позиций могут быть "верными" семантически
- При коротких предложениях случайный baseline даст неплохой скор → конкуренты тоже быстро стартуют

---

## ⚖️ Итоговое сравнение

| Критерий | Case 1: Пунктуация | Case 2: Пропущенные слова |
|---|:---:|:---:|
| **Время до первого сабмита** | 2–3 часа | **< 30 минут** ⭐ |
| **Готовый трейн** | ❌ Строишь сам | ✅ 550к готово ⭐ |
| **Метрика** | Macro F1 (сложнее) | Accuracy (прозрачно) ⭐ |
| **Конкурентность** | Высокая | Средняя ⭐ |
| **LLM API запрещён** | ✅ (для обоих) | ✅ (для обоих) |
| **Питчинг** | Хороший | **Отличный** ⭐ |
| **Потолок улучшений** | Высокий | Высокий |
| **Главный риск** | Построить трейн | Строгость Accuracy |

---

## 🎯 Финальная рекомендация

### 🥇 **Case 2 (Пропущенные слова)** — для большинства команд

**Три главные причины:**
1. **550к трейна** — тратишь время на модель, а не на сборку данных
2. **30 минут до первого сабмита** — больше итераций за хакатон
3. **Меньше конкурентов** — реальнее попасть в Top-20

### 🥈 **Case 1 (Пунктуация)** — если команда сильна в NLP
- Есть готовый код для token-classification
- Готовы сразу качать multidomain-kazakh-dataset и строить пайплайн
- Хотите максимальный технический скор, а не просто Top-20

---

## 📋 Чеклист к дедлайну

- [ ] Заполнить форму выбора кейса → промокод Yandex Cloud (10 000 ₽)
- [ ] Убедиться что inference воспроизводится в **Kaggle Notebook**
- [ ] Загрузить предсказания до **12:00, 15 марта**
- [ ] Подготовить 2 лучших Kaggle Notebook (только inference-код)
- [ ] Презентация до **13:00** (рек.) / **13:30** (дедлайн)
- [ ] Выступление — до **7 минут**
- [ ] **Не использовать OpenAI/Anthropic API** для генерации предсказаний ❌

---

*WISH Hackathon 2026 · Удачи! 🚀*
