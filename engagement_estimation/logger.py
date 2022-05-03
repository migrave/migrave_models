from termcolor import colored

class Logger(object):
    """Defines convenience functions for logging information to the terminal.
    All functions are defined as static.
    """

    @staticmethod
    def info(msg):
        """Prints an info message to the screen.

        Keyword arguments:
        msg: str -- message to log

        """
        print(colored('[INFO] ' + msg, color='blue'))

    @staticmethod
    def warning(msg: str) -> None:
        """Prints a warning message to the screen (coloured yellow).

        Keyword arguments:
        msg: str -- message to log

        """
        print(colored('[WARNING] ' + msg, color='yellow'))

    @staticmethod
    def error(msg):
        """Prints an error message to the screen (coloured red).

        Keyword arguments:
        msg: str -- message to log

        """
        print(colored('[ERROR] ' + msg, color='red'))
