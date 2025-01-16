import logging

class LoggerWrapper():

    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger(str(__package__))
        self.logger.setLevel(level)
        handlerFile = logging.FileHandler("log-pruning", mode='w')
        handlerConsole = logging.StreamHandler()
        self.level = level

        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

        handlerFile.setFormatter(formatter)
        handlerConsole.setFormatter(formatter)

        self.logger.addHandler(handlerFile)
        self.logger.addHandler(handlerConsole)

    def __call__(self, message):
        self.logger.log(self.level, message)
