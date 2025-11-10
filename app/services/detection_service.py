from pathlib import Path

from loguru import logger

from app.integrations.telegram_notifier import TelegramNotifier, TelegramAlertData
from app.integrations.yolo_detector import YoloDetector, DetectionResult


class DetectionService:
    """
    Сервис обработки детекции огня и дыма.
    Отвечает за бизнес-логику: анализ результатов YOLO и принятие решения о тревоге.
    """

    def __init__(
            self,
            yolo_detector: YoloDetector,
            telegram_notifier: TelegramNotifier,
            fire_confidence_threshold: float = 0.7,
            smoke_confidence_threshold: float = 0.65,
    ):
        """
        Инициализация сервиса.

        Args:
            yolo_detector (YoloDetector): Экземпляр детектора YOLOv11.
            telegram_notifier (TelegramNotifier): Экземпляр отправителя уведомлений.
            fire_confidence_threshold (float): Порог уверенности для срабатывания тревоги по огню.
            smoke_confidence_threshold (float): Порог уверенности для срабатывания тревоги по дыму.
        """
        self.yolo_detector = yolo_detector
        self.telegram_notifier = telegram_notifier
        self.fire_threshold = fire_confidence_threshold
        self.smoke_threshold = smoke_confidence_threshold

    async def analyze_image(self, image_path: str | Path) -> dict:
        """
        Анализирует изображение на наличие огня или дыма.

        Args:
            image_path (str | Path): Путь к изображению для анализа.

        Returns:
            dict: Результат анализа с информацией о тревоге и статусе.
        """
        logger.info(f"Начало анализа изображения: {image_path}")

        # 1. Выполняем инференс
        detection_result: DetectionResult = await self.yolo_detector.predict(image_path)

        # 2. Анализируем результаты
        has_fire = False
        has_smoke = False
        fire_detections = []
        smoke_detections = []

        for detection in detection_result.detections:
            if detection.class_name == "Fire" and detection.confidence >= self.fire_threshold:
                has_fire = True
                fire_detections.append(detection)
            elif detection.class_name == "Smoke" and detection.confidence >= self.smoke_threshold:
                has_smoke = True
                smoke_detections.append(detection)

        # 3. Принятие решения: если есть огонь — это критическая тревога.
        #    Если есть дым — это предупреждение.
        alert_level = "INFO"
        alert_text = "Визуальная проверка завершена. Угроз не обнаружено."

        if has_fire:
            alert_level = "CRITICAL"
            alert_text = (
                f"🚨 **КРИТИЧЕСКАЯ УГРОЗА: ОГОНЬ ОБНАРУЖЕН!** 🚨\n"
                f"Обнаружено {len(fire_detections)} очагов огня.\n"
                f"Уверенность: {max(d.confidence for d in fire_detections):.2%}"
            )
        elif has_smoke:
            alert_level = "WARNING"
            alert_text = (
                f"⚠️ **ПРЕДУПРЕЖДЕНИЕ: ДЫМ ОБНАРУЖЕН!** ⚠️\n"
                f"Обнаружено {len(smoke_detections)} областей дыма.\n"
                f"Уверенность: {max(d.confidence for d in smoke_detections):.2%}"
            )

        # 4. Формируем данные для уведомления
        alert_data = TelegramAlertData(
            message_text=alert_text,
            image_path=image_path if detection_result.success else None
        )

        # 5. Отправляем уведомление
        sent_success = await self.telegram_notifier.send_alert(alert_data)

        # 6. Возвращаем результат
        result = {
            "success": detection_result.success,
            "alert_level": alert_level,
            "message": alert_text,
            "has_fire": has_fire,
            "has_smoke": has_smoke,
            "fire_count": len(fire_detections),
            "smoke_count": len(smoke_detections),
            "notification_sent": sent_success,
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": d.confidence,
                    "x1": d.x1,
                    "y1": d.y1,
                    "x2": d.x2,
                    "y2": d.y2
                } for d in detection_result.detections
            ]
        }

        logger.info(f"Анализ завершен. Результат: {result}")
        return result
