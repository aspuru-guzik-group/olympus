#!/usr/bin/env python

# ==============================================================================

from olympus import Logger

from .connector_server import ConnectorServer
from .connector_github import ConnectorGithub

# ==============================================================================


class ParserBaseline:
    def __init__(self, subparsers):
        self.parser = subparsers.add_parser("baseline", help=">> help for baseline")
        self.parser.add_argument("--get", dest="get", action="store_true")
        self.parser.set_defaults(get=False)

    @staticmethod
    def __call__(args):
        get = args.get

        if get is True:

            for Connector in [ConnectorGithub, ConnectorServer]:
                success = Connector().get_baseline()
                if success:
                    break

        elif get is False:
            pass
