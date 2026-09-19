import logging
import os
import sys

from loguru import logger


class Settings:
    """Configuracion de la API. Se puede sobreescribir con variables de entorno."""

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "HotelRisk API")
    LOGGING_LEVEL: int = int(os.getenv("LOGGING_LEVEL", logging.INFO))


class InterceptHandler(logging.Handler):
    """Redirige los mensajes de logging estandar (uvicorn) a loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_app_logging(config: Settings) -> None:
    logging.getLogger().handlers = [InterceptHandler()]
    for nombre in ("uvicorn.asgi", "uvicorn.access"):
        logging.getLogger(nombre).handlers = [InterceptHandler(level=config.LOGGING_LEVEL)]
    logger.configure(handlers=[{"sink": sys.stderr, "level": config.LOGGING_LEVEL}])


settings = Settings()
