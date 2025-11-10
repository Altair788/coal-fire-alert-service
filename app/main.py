from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.api.v1.endpoints.detection import detection_service
from app.api.v1.endpoints.detection import router as detection_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск приложения. Инициализация ресурсов...")

    logger.info("Приложение запущено.")
    yield
    logger.info("Остановка приложения. Закрытие ресурсов...")
    await detection_service.telegram_notifier.close()
    logger.info("Ресурсы закрыты. Приложение остановлено.")


app = FastAPI(
    title="Coal Fire Alert Service",
    description="API для обнаружения признаков самовозгорания угля с помощью YOLOv11 и уведомлений через Telegram.",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(detection_router, prefix="/api/v1", tags=["detection"])


@app.get("/")
def read_root():
    return {"message": "Coal Fire Alert Service is running!"}

# Запуск: uvicorn app.main:app --reload
