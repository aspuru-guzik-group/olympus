#!/usr/bin/env python

import os

from olympus import Logger, __home__

from .connector_github import ConnectorGithub
from .connector_server import ConnectorServer

# ===============================================================================


# ===============================================================================


class ParserDownload:
    def __init__(self, subparsers):
        self.parser = subparsers.add_parser(
            "download", help=">> help for download"
        )
        self.group = self.parser.add_mutually_exclusive_group(required=True)
        self.group.add_argument(
            "-n",
            "--name",
            action="store",
        )
        self.group.add_argument(
            "-l",
            "--list",
            action="store_true",
        )

    @staticmethod
    def __call__(args):
        if args.list is True:
            ParserDownload._list_datasets()
        elif args.name is not None:
            ParserDownload._get_dataset(args.name)
        else:
            Logger.log("could not parse command line arguments", "ERROR")

    @staticmethod
    def _list_datasets():
        # check github first and the server only as a backup
        for Connector in [ConnectorGithub, ConnectorServer]:
            datasets = Connector().list()
            if isinstance(datasets, list):
                Logger.log(f"found datasets: {datasets}", "INFO")
                return
        Logger.log("could not retrieve list of datasets", "ERROR")

    @staticmethod
    def _get_dataset(dataset_name):

        # check if dataset already exists
        expected_files = ["dataset.zip", "data.csv", "description.txt"]
        exists = False
        for expected_file in expected_files:
            exists = exists or os.path.isfile(
                f"{__home__}/datasets/dataset_{dataset_name}/{expected_file}"
            )
        if exists:
            Logger.log("The dataset already exists", "INFO")
            return

        # download dataset, check with github first
        for Connector in [ConnectorGithub, ConnectorServer]:
            success = Connector().get_dataset(dataset_name)
            if success:
                break


# ===============================================================================
