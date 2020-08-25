import logging
import logging.config
import os

from typing import Union


class Reporter:

    def __init__(self,
                 logger: Union[logging.Logger, None] = None,
                 lvl: int = logging.INFO) -> None:

        """Reporter Class.

        Parameters
        ----------
        logger : logger or None
            The logger object to use.
        lvl : int
            Logging level to write to.
        """

        self.logger = logger or logging.getLogger(__name__)
        self.log_level = lvl
        self.log_path = os.path.join(os.getcwd(), 'report.log')

        self._setup_logger()

    def log(self,
            msg: str,
            lvl: int = logging.INFO) -> None:

        """Logs a message with a given level.

        Parameters
        ----------
        msg : str
            Message to log.
        lvl : int
            Level with which to log the message.
        """

        self.logger.log(lvl, msg)

    def _setup_logger(self) -> None:

        """Initialises the logger using the basic config."""

        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%d/%m/%Y %H:%M:%S',
            filename=self.log_path,
            filemode='w'
        )
