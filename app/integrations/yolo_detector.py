import asyncio
from pathlib import Path
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel
from ultralytics import YOLO


class DetectionBox(BaseModel):
    """Pydantic модель для одной обнаруженной bounding box."""
    class_name: str
    confidence: float
    # x1, y1, x2, y2 - координаты bbox (в пикселях или нормализованные)
    # Для простоты пока без координат, добавим позже, если нужно
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    """Pydantic модель для результата детекции для одного изображения."""
    success: bool
    message: str
    detections: List[DetectionBox] = []
    image_path: Optional[str] = None


class YoloDetector:
    """
    Модуль интеграции с YOLOv11 моделью для обнаружения огня и дыма.
    Следует принципу SRP: только загрузка модели и выполнение инференса.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.35, iou_threshold: float = 0.1):
        """
        Инициализирует YoloDetector.

        Args:
            model_path (str): Путь к файлу .pt модели YOLOv11.
            confidence_threshold (float): Порог уверенности для детекции.
            iou_threshold (float): Порог IoU для подавления немаксимумов.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model: Optional[YOLO] = None
        self._lock = asyncio.Lock()  # Для потокобезопасной инициализации

    async def load_model(self):
        """
        Асинхронно загружает модель YOLOv11.
        Использует Lock, чтобы избежать проблем при параллельной загрузке.
        """
        async with self._lock:
            if self._model is None:
                logger.info(f"Загрузка модели YOLOv11 из {self.model_path}...")
                try:
                    # Загружаем модель
                    self._model = YOLO(self.model_path)
                    logger.success("Модель YOLOv11 успешно загружена.")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке модели: {e}")
                    raise RuntimeError(f"Не удалось загрузить модель YOLOv11 из {self.model_path}") from e

    async def predict(self, source: str | Path) -> DetectionResult:
        """
        Выполняет инференс модели на изображении или видеофайле.

        Args:
            source (str | Path): Путь к изображению или видеофайлу, или URL-адрес потока.

        Returns:
            DetectionResult: Результат детекции.
        """
        if self._model is None:
            await self.load_model()

        try:
            logger.info(f"Выполнение инференса на источнике: {source}")

            # Выполняем предсказание
            # conf - порог уверенности, iou - порог IoU для NMS
            results = self._model(
                source=source,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False  # Отключаем подробный вывод ultralytics в консоль
            )

            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])  # Уверенность
                        class_name = r.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        detections.append(
                            DetectionBox(
                                class_name=class_name,
                                confidence=conf,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2
                            )
                        )

            logger.info(f"Обнаружено {len(detections)} объектов в {source}")
            return DetectionResult(
                success=True,
                message=f"Инференс успешно выполнен. Обнаружено {len(detections)} объектов.",
                detections=detections,
                image_path=str(source) if isinstance(source, (str, Path)) else None
            )

        except Exception as e:
            logger.error(f"Ошибка при выполнении инференса на {source}: {e}")
            return DetectionResult(
                success=False,
                message=f"Ошибка инференса: {str(e)}",
                detections=[],
                image_path=str(source) if isinstance(source, (str, Path)) else None
            )
