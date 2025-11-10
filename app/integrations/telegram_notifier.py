import asyncio
from pathlib import Path

from aiogram import Bot
from aiogram.types import BufferedInputFile, Message
from loguru import logger
from pydantic import BaseModel


class TelegramAlertData(BaseModel):
    """Данные для отправки в Telegram."""
    message_text: str
    image_path: str | Path | None = None


class TelegramNotifier:
    """
    Модуль интеграции с Telegram для отправки уведомлений.
    """

    def __init__(self, bot_token: str, chat_id: str):
        """
        Args:
            bot_token (str): Токен Telegram-бота.
            chat_id (str): ID чата, в который будут отправляться уведомления.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot: Bot | None = None
        self._lock = asyncio.Lock()  # Для потокобезопасной инициализации бота

    async def _ensure_bot(self):
        """
        Асинхронно создает и сохраняет экземпляр бота, если он еще не создан.
        Использует Lock, чтобы избежать проблем при параллельной инициализации.
        """
        async with self._lock:
            if self._bot is None:
                logger.info("Создание экземпляра Telegram Bot...")
                self._bot = Bot(token=self.bot_token)
                logger.success("Экземпляр Telegram Bot успешно создан.")

    async def send_alert(self, alert_data: TelegramAlertData) -> bool:
        """
        Отправляет уведомление в Telegram.

        Args:
            alert_data (TelegramAlertData): Данные для отправки (текст, изображение).

        Returns:
            bool: True, если сообщение отправлено успешно, иначе False.
        """
        try:
            await self._ensure_bot()

            text = alert_data.message_text

            photo = None
            if alert_data.image_path:
                image_path = Path(alert_data.image_path)
                if image_path.exists() and image_path.is_file():
                    photo_bytes = image_path.read_bytes()

                    photo = BufferedInputFile(photo_bytes, filename=image_path.name)
                    logger.info(f"Подготовлено изображение для отправки: {image_path}")
                else:
                    logger.warning(f"Файл изображения не найден или не является файлом: {image_path}")
                    sent_message: Message = await self._bot.send_message(
                        chat_id=self.chat_id,
                        text=text
                    )
                    logger.info(f"Текстовое уведомление отправлено в чат {self.chat_id}")
                    return True

            if photo:
                sent_message: Message = await self._bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=text
                )
                logger.success(f"Фото с уведомлением отправлено в чат {self.chat_id}")
            else:
                sent_message: Message = await self._bot.send_message(
                    chat_id=self.chat_id,
                    text=text
                )
                logger.success(f"Текстовое уведомление отправлено в чат {self.chat_id}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при отправке уведомления в Telegram: {e}")
            return False

    async def close(self):
        if self._bot:
            await self._bot.session.close()
            logger.info("Соединение с Telegram Bot API закрыто.")
