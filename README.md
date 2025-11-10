# Coal Fire Alert Service  
**Прогноз самовозгорания угля на открытых складах**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)  
![Asyncio usage](https://img.shields.io/badge/asyncio-enabled-brightgreen)  
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![YOLOv11](https://img.shields.io/badge/detect-YOLOv11-FF6B35)   
![aiogram](https://img.shields.io/badge/Telegram-aiogram-25A2E0)
---

## Описание проекта

**Coal Fire Alert Service** — асинхронный FastAPI-сервис для **раннего обнаружения признаков самовозгорания угля** на открытых площадках с использованием **YOLOv11** и мгновенной отправки оповещений через **Telegram**.

Проект создан как прототип для хакатона и демонстрирует **промышленное применение компьютерного зрения** в сфере угледобычи и логистики. Вместо традиционных датчиков температуры, сервис анализирует видеопоток (или статичные изображения) с камер, обнаруживает **огонь** и **дым**, и в случае обнаружения — немедленно отправляет **визуальное оповещение** с изображением и деталями в Telegram-чат.

Реализация основана на готовой модели `best_nano_111.pt` из репозитория [Flare Guard](https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11), адаптированной под промышленные задачи.

---

## Основные возможности

- ✅ **Обнаружение огня и дыма** с помощью YOLOv11 nano-модели (модель обучена на 10K+ изображений).
- ✅ **Асинхронная обработка** изображений через FastAPI и `ultralytics`.
- ✅ **Мгновенные уведомления** в Telegram с прикреплённым изображением и деталями (уровень уверенности, тип угрозы).
- ✅ **Гибкая логика срабатывания**:  
  - `CRITICAL` — обнаружен огонь с уверенностью > 70%  
  - `WARNING` — обнаружен дым с уверенностью > 65%

---

## Интеграции

- **YOLOv11 (ultralytics)** — модель для обнаружения объектов `Fire` и `Smoke`.  
- **Telegram (aiogram)** — система мгновенных уведомлений с изображениями.  
- **FastAPI** — REST API для приёма изображений и возврата результатов.

> ✅ **Расширяемость**: Легко добавить новые источники данных (например, температурные сенсоры), новые каналы оповещения (WhatsApp, SMTP, amoCRM) или другие модели (YOLOv8, EfficientDet).

---

## Зависимости

### Основные зависимости:

- `fastapi` — асинхронный веб-фреймворк
- `ultralytics` — реализация YOLOv11
- `aiogram` — асинхронный Telegram-бот
- `pydantic` — валидация и сериализация данных
- `environs` — управление переменными окружения
- `loguru` — продвинутое логирование
- `uvicorn` — ASGI-сервер для запуска FastAPI
- `python-multipart` — обработка загрузки файлов

### Dev-зависимости:

- `isort`, `black`, `ruff` — автоформатирование и линтинг
- `pytest`, `pytest-asyncio` — тестирование

> Полный список с версиями — в `pyproject.toml`.  
> Установка через **Poetry**:

```bash
poetry install
```

---

## Установка и запуск

### 1. Установка зависимостей

Убедитесь, что установлены:
- Python 3.12+
- [Poetry](https://python-poetry.org/)

```bash
poetry install
```

### 2. Настройка переменных окружения

Скопируйте шаблон и заполните свои данные:

```bash
cp .env.example .env
```

Откройте `.env` и задайте:

```env
# --- YOLO ---
YOLO_MODEL_PATH=models/best_nano_111.pt
YOLO_CONFIDENCE_THRESHOLD=0.7
YOLO_IOU_THRESHOLD=0.65

# --- Telegram ---
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=-100xxxxxxxxx  # ID чата, куда бот должен отправлять уведомления

# --- FastAPI ---
APP_HOST=0.0.0.0
APP_PORT=8000
```

### 4. Запуск сервера

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Сервер запустится на:  
👉 [http://localhost:8000](http://localhost:8000)

---

## Использование API

### 📸 Отправить изображение для анализа

Используйте `curl` или Postman:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@test_images/fire_example.png"
```

✅ **Пример ответа** (если обнаружен огонь):

```json
{
  "success": true,
  "alert_level": "CRITICAL",
  "message": "🚨 **КРИТИЧЕСКАЯ УГРОЗА: ОГОНЬ ОБНАРУЖЕН!** 🚨\nОбнаружено 1 очагов огня.\nУверенность: 83.49%",
  "has_fire": true,
  "has_smoke": false,
  "fire_count": 1,
  "smoke_count": 0,
  "notification_sent": true,
  "detections": [
    {
      "class": "Fire",
      "confidence": 0.8349120020866394,
      "x1": 172.0371551513672,
      "y1": 1.671006441116333,
      "x2": 770.7168579101562,
      "y2": 336.2496032714844
    }
  ]
}
```

> 🔔 **Уведомление автоматически приходит в Telegram** с изображением и текстом.

### 📚 Документация API

После запуска сервера откройте в браузере:  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

Там вы найдёте:
- Интерактивную документацию OpenAPI
- Возможность отправить файл через кнопку “Try it out”

---

## Структура проекта

```text
coal-fire-alert-service/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Точка входа FastAPI-приложения
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── health.py      # /health
│   │           └── detection.py   # /api/v1/analyze (основной эндпоинт)
│   ├── core/
│   │   └── settings.py            # Настройки (environs)
│   ├── integrations/
│   │   ├── yolo_detector.py       # YOLOv11 инференс (SRP)
│   │   └── telegram_notifier.py   # Отправка уведомлений (SRP)
│   ├── services/
│   │   └── detection_service.py   # Бизнес-логика: анализ → решение → уведомление (SRP)
│   ├── schemas/
│   │   ├── request.py             # Pydantic: ImageUpload
│   │   └── response.py            # Pydantic: DetectionResult
│   └── tests/
│       ├── __init__.py
│       ├── test_detector.py
│       └── test_service.py
├── models/
│   └── best_nano_111.pt           # Обученная модель YOLOv11
├── test_images/
│   └── fire_example.png           # Пример для тестирования
├── .env.example                   # Шаблон переменных окружения
├── pyproject.toml                 # Зависимости и настройки Poetry
├── poetry.lock                    # Фиксированные версии
├── README.md
└── .gitignore
```

---


### 🧪 Запуск в режиме отладки

```bash
# Запуск без авто-перезагрузки (для отладки)
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Запуск с логами уровня DEBUG
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

---

## Важные замечания

1. **Только изображения**: Сервис сейчас работает с **статичными изображениями**.
---

## Авторство

Проект разработан **[Eduard Slobodyanik](mailto:slobodyanik.ed@gmail.com)**
в рамках подготовки к хакатону по теме "Прогноз самовозгорания угля".

> 🚀 *Предназначен для демонстрации профессионального подхода к инженерии, а не только к ML.*

---

## 🛡️ Protect What Matters Most — Early Detection Saves Lives

