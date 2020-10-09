#!/usr/bin/env python

# ======================================================================

import sys
import traceback

# ======================================================================


class MessageLogger:

    VERBOSITY_LEVELS = {
        -1: [],
        0: ["FATAL"],
        1: ["INFO", "FATAL"],
        2: ["INFO", "ERROR", "FATAL"],
        3: ["INFO", "WARNING", "ERROR", "FATAL"],
        4: ["DEBUG", "INFO", "WARNING", "ERROR", "FATAL"],
    }

    WRITER = {
        "DEBUG": sys.stdout,
        "INFO": sys.stdout,
        "WARNING": sys.stderr,
        "ERROR": sys.stderr,
        "FATAL": sys.stderr,
    }

    # more colors and styles:
    # https://stackoverflow.com/questions/2048509/how-to-echo-with-different-colors-in-the-windows-command-line
    # https://joshtronic.com/2013/09/02/how-to-use-colors-in-command-line-output/

    GREY = "0;37"
    WHITE = "1;37"
    YELLOW = ("1;33",)
    LIGHT_RED = ("1;31",)
    RED = "0;31"
    COLORS = {
        "DEBUG": WHITE,
        "INFO": GREY,
        "WARNING": YELLOW,
        "ERROR": LIGHT_RED,
        "FATAL": RED,
    }

    def __init__(self, verbosity=4):
        self.set_verbosity(verbosity)
        for key in self.WRITER:
            setattr(self, "{}S".format(key), [])

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity
        self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]

    def log(self, message, message_type, only_once=False):

        if only_once:
            logs = getattr(self, f"{message_type}S")
            if message in logs:
                return

        logs = getattr(self, "{}S".format(message_type))
        logs.append(message)
        setattr(self, "{}S".format(message_type), logs)

        if only_once:
            message += "\n    [This message will be shown only once]"

        if not message_type in self.verbosity_levels:
            return None
        color = self.COLORS[message_type]
        writer = self.WRITER[message_type]
        if message_type in ["WARNING", "ERROR", "FATAL"]:
            error = traceback.format_exc()
            if error != "NoneType: None\n":
                writer.write(error)
        uncolored = "[{message_type}] {message}\n".format(
            message_type=message_type, message=message
        )
        message = "\x1b[%sm" % (color) + uncolored + "\x1b[0m"
        writer.write(message)

        if message_type == "FATAL":
            raise SystemExit

    def purge(self):
        for message_type in self.WRITER.keys():
            setattr(self, "{}S".format(message_type), [])


# ======================================================================
