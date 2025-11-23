import sys
import logging

from datetime import datetime
from pathlib import Path
from termcolor import colored


__all__ = ['init_logger', 'info', 'error', 'warn', 'debug', 'exception']


logger = logging.getLogger("gateway")


def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs, stacklevel=2)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, stacklevel=2)

def warn(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs, stacklevel=2)

def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs, stacklevel=2)

def exception(msg, *args, **kwargs):
    logger.exception(msg, *args, **kwargs, stacklevel=2)


FMT = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d %(funcName)s] %(message)s'
DATE_FMT = '%Y%m%d %H:%M:%S'

LEVEL_COLORS = {
    logging.ERROR: 'red',
    logging.WARNING: 'yellow'
}


class DailyFileHandler(logging.Handler):
    def __init__(self, log_dir, encoding='utf-8'):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.encoding = encoding
        self.current_date = None
        self.file_handler: logging.FileHandler | None = None
        self._update_file_handler()

    def _update_file_handler(self):
        today = datetime.now().strftime("%Y%m%d")

        if today != self.current_date or self.file_handler is None:
            if self.file_handler:
                self.file_handler.close()

            log_file_path = self.log_dir / f"{today}.log"
            self.file_handler = logging.FileHandler(
                filename=log_file_path,
                encoding=self.encoding
            )
            self.file_handler.setFormatter(self.formatter)
            self.current_date = today

    def emit(self, record: logging.LogRecord):
        self._update_file_handler()
        if self.file_handler:
            self.file_handler.emit(record)

    def setFormatter(self, fmt: logging.Formatter | None):
        super().setFormatter(fmt)
        if self.file_handler:
            self.file_handler.setFormatter(fmt)


def init_logger(
        logs_dir: Path
) -> None:
    class ColoredConsoleHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            original_levelname = record.levelname

            if record.levelno in LEVEL_COLORS:
                record.levelname = colored(record.levelname, LEVEL_COLORS[record.levelno])

            log_entry = self.format(record)
            record.levelname = original_levelname

            sys.stderr.write(f"{log_entry}\n")
            sys.stderr.flush()

    logs_dir.mkdir(parents=True, exist_ok=True)

    file_handler = DailyFileHandler(logs_dir, encoding="utf-8")

    file_handler.setFormatter(logging.Formatter(
        FMT, DATE_FMT
    ))

    console_handler = ColoredConsoleHandler()

    logging.basicConfig(
        level=logging.INFO,
        format=FMT,
        datefmt=DATE_FMT,
        handlers=[console_handler, file_handler]
    )
