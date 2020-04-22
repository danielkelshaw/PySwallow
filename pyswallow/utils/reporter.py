import logging
import logging.config
import os


class Reporter:

    def __init__(self, logger=None, lvl=logging.INFO):

        self.logger = logger or logging.getLogger(__name__)
        self.log_level = lvl
        self.log_path = os.path.join(os.getcwd(), 'report.log')

        self._setup_logger()

    def log(self, msg, lvl=logging.INFO):
        self.logger.log(lvl, msg)

    def _setup_logger(self):
        logging.basicConfig(level=self.log_level,
                            format='%(asctime)s '
                                   '%(name)-12s '
                                   '%(levelname)-8s '
                                   '%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.log_path,
                            filemode='w')
