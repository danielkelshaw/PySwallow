import logging
import logging.config
import os


class Reporter:

    def __init__(self, logger=None, lvl=logging.INFO):

        """
        Initialiser for Reporter class.

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

    def log(self, msg, lvl=logging.INFO):

        """
        Logs a message with a given level.

        Parameters
        ----------
        msg : str
            Message to log.
        lvl : int
            Level with which to log the message.
        """

        self.logger.log(lvl, msg)

    def _setup_logger(self):

        """Initialises the logger using the basic config."""

        logging.basicConfig(level=self.log_level,
                            format='%(asctime)s '
                                   '%(name)-12s '
                                   '%(levelname)-8s '
                                   '%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.log_path,
                            filemode='w')
