from environs import Env

env = Env()
env.read_env()

class Settings:
    # --- YOLO ---
    YOLO_MODEL_PATH: str = env.str("YOLO_MODEL_PATH", "models/best_nano_111.pt")
    YOLO_CONFIDENCE_THRESHOLD: float = env.float("YOLO_CONFIDENCE_THRESHOLD", 0.35) # Порог уверенности
    YOLO_IOU_THRESHOLD: float = env.float("YOLO_IOU_THRESHOLD", 0.1) # Порог IoU

    # --- Telegram ---
    TELEGRAM_BOT_TOKEN: str = env.str("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = env.str("TELEGRAM_CHAT_ID", "")

    # --- FastAPI ---
    APP_HOST: str = env.str("APP_HOST", "0.0.0.0")
    APP_PORT: int = env.int("APP_PORT", 8000)

settings = Settings()