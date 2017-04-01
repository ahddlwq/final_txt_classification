import logging
from logging.handlers import TimedRotatingFileHandler

from config.config import FilePathConfig


class LogTool(object):
    log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
    formatter = logging.Formatter(log_fmt)
    log_file_handler = TimedRotatingFileHandler(filename=FilePathConfig.log_path, when="D", interval=1,
                                                backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d.log"
    log_file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG)

    def __init__(self):
        self.log = logging.getLogger()
        self.log.addHandler(LogTool.log_file_handler)
