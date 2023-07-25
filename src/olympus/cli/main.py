#!/usr/bin/env python

# simple interface to the available scripts
import sys
from argparse import SUPPRESS, ArgumentParser, RawTextHelpFormatter

from olympus import __version__


class OlympusCli:
    def __init__(self):
        self.parser = ArgumentParser(
            description="""
    ------------------------------
    Olympus command line interface
    ------------------------------
    Available commands are:
        baseline     download baseline results with random search
        download     downloads dataset from olympus_datasets
        upload       uploads dataset to olympus_datasets
        etc          Etc""",
            formatter_class=RawTextHelpFormatter,
        )

        self.parser.add_argument(
            "-v", "--version", action="version", version=__version__
        )

        self._register_subparsers()
        self._parse_args()

    def _register_subparsers(self):
        self.subparsers = self.parser.add_subparsers(
            help="subparser help", dest="subparser_name"
        )

        from .cli_baseline import ParserBaseline
        from .cli_download import ParserDownload
        from .cli_upload import ParserUpload

        self.parser_baseline = ParserBaseline(self.subparsers)
        self.parser_download = ParserDownload(self.subparsers)
        self.parser_upload = ParserUpload(self.subparsers)

    def _parse_args(self):

        # first parse
        args, rest = self.parser.parse_known_args()
        if args.subparser_name is None:
            self.parser.print_help()
            return

        subparser = getattr(self, f"parser_{args.subparser_name}")
        success = subparser(args)
        # subsequent parse
        while len(rest) > 0 and success:
            args, rest = self.parser.parse_known_args(rest)
            subparser = getattr(self, f"parser_{args.subparser_name}")
            success = subparser(args)


def check_unknown_cmd(unknowns):
    """Checks unknown command line arguments are raises a warning if unexpected
    commands are found.
    """
    expected = ["olympus", "get"]

    for cmd in unknowns:
        if cmd not in expected:
            print(
                'Unknown command found in your command line: "{}". '
                "This command will be ignored".format(cmd)
            )


def entry_point():
    OlympusCli()


if __name__ == "__main__":
    entry_point()
