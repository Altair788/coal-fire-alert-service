import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.settings import settings
from app.integrations.telegram_notifier import TelegramNotifier
from app.integrations.yolo_detector import YoloDetector
from app.services.detection_service import DetectionService

router = APIRouter()

yolo_detector = YoloDetector(
    model_path=settings.YOLO_MODEL_PATH,
    confidence_threshold=settings.YOLO_CONFIDENCE_THRESHOLD,
    iou_threshold=settings.YOLO_IOU_THRESHOLD
)

telegram_notifier = TelegramNotifier(
    bot_token=settings.TELEGRAM_BOT_TOKEN,
    chat_id=settings.TELEGRAM_CHAT_ID
)

detection_service = DetectionService(
    yolo_detector=yolo_detector,
    telegram_notifier=telegram_notifier
)


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Принимает изображение и возвращает результат детекции.
    """
    # Сохраняем файл во временную папку
    temp_path = Path(f"temp/{uuid.uuid4().hex}_{file.filename}")
    temp_path.parent.mkdir(exist_ok=True)

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Анализируем
        result = await detection_service.analyze_image(temp_path)

        # Удаляем временный файл
        temp_path.unlink()

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")
