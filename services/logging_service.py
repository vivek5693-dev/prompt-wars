import logging
import os
from typing import Any

class LoggingService:
    """Service for structured Google Cloud Logging."""
    
    @staticmethod
    def initialize() -> None:
        """Initializes structured logging via Google Cloud Logging if available."""
        try:
            import google.cloud.logging
            client = google.cloud.logging.Client()
            client.setup_logging()
            logging.info("Google Cloud Logging successfully initialized.")
        except Exception as e:
            logging.basicConfig(level=logging.INFO)
            logging.warning(f"Cloud Logging fallback to basic logging: {e}")

    @staticmethod
    def info(message: str, **kwargs: Any) -> None:
        logging.info(message, extra=kwargs)

    @staticmethod
    def error(message: str, **kwargs: Any) -> None:
        logging.error(message, extra=kwargs)

    @staticmethod
    def warning(message: str, **kwargs: Any) -> None:
        logging.warning(message, extra=kwargs)

    @staticmethod
    def critical(message: str, **kwargs: Any) -> None:
        logging.critical(message, extra=kwargs)
