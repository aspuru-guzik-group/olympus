#!/usr/bin/env python

import glob
import os
import pickle

from olympus.databases import Database
from olympus import Logger
from olympus.objects import Object
from olympus.datasets import list_datasets

__home__ = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================


class Baseline(Object):
    def __init__(self, summary_file=f"{__home__}/baseline_summary.pkl"):
        Object.__init__(**locals())
        self._load_baselines()

    def _load_summaries(self, datasets_):
        datasets = datasets_.copy()
        self.baseline_summaries = {}
        # nothing to do if the summary is not available
        if not os.path.isfile(self.summary_file):
            return
        # load summary
        with open(self.summary_file, "rb") as content:
            baseline_summaries = pickle.load(content)
        for dataset, summary in baseline_summaries.items():
            if dataset in datasets:
                self.baseline_summaries[dataset] = summary
                datasets.remove(dataset)
            else:
                Logger.log(
                    f"found summary for not reported dataset: {dataset}",
                    "WARNING",
                )
        for dataset in datasets:
            Logger.log(
                f"could not find summary for dataset: {dataset}", "WARNING"
            )

    def _register_dbs(self, datasets_):
        # only register complete baselines
        datasets = datasets_.copy()
        self.baseline_db_files = {}
        self.baseline_dbs = {}
        for db_file in glob.glob(f"{__home__}/db_baseline_*sqlite"):
            dataset = db_file.split("/")[-1].split(".")[0][12:]
            self.baseline_db_files[dataset] = db_file
            if dataset in datasets:
                datasets.remove(dataset)
            else:
                Logger.log(
                    f"found complete baseline for not reported dataset: {dataset}",
                    "WARNING",
                )
        for dataset in datasets:
            Logger.log(
                f"could not find complete baseline for dataset: {dataset}",
                "WARNING",
            )

    def _load_baseline_db(self, dataset):
        file_name = self.baseline_db_files[dataset]
        db = Database().from_file(file_name)
        self.baseline_dbs[dataset] = db
        return db

    def _load_baselines(self):
        # get list of datasets for which we expect baseline
        datasets = list_datasets()
        # load summaries of baseline
        self._load_summaries(datasets)
        # get file names and paths for complete baselines
        self._register_dbs(datasets)

    def get_summary(self, dataset):
        if dataset in self.baseline_summaries.keys():
            return self.baseline_summaries[dataset]
        else:
            Logger.log(
                f"could not find summary for dataset: {dataset}", "ERROR"
            )

    def get_db(self, dataset):
        if dataset in self.baseline_dbs.keys():
            return self.baseline_dbs[dataset]
        elif dataset in self.baseline_db_files.keys():
            self._load_baseline_db(dataset)
            return self.baseline_dbs[dataset]
        else:
            Logger.log(
                f"could not find baseline db for dataset: {dataset}", "ERROR"
            )

    def get_campaigns(self, dataset):
        if dataset in self.baseline_dbs.keys():
            return [campaign for campaign in self.baseline_dbs[dataset]]
        elif dataset in self.baseline_db_files.keys():
            self._load_baseline_db(dataset)
            return [campaign for campaign in self.baseline_dbs[dataset]]
        else:
            Logger.log(
                f"could not find baseline db for dataset: {dataset}", "ERROR"
            )

    def get(self, dataset, kind="summary"):
        """Retrieves baseline for a given dataset

        Args:
            dataset (str): name of the dataset for which baseline should be retrieved
            kind (str): indicates format of baseline; choose from "summary", "db", or "campaigns"

        Returns:
            requested baseline
        """
        if kind == "summary":
            return self.get_summary(dataset)
        elif kind == "db":
            return self.get_db(dataset)
        elif kind == "campaigns":
            return self.get_campaigns(dataset)
        else:
            Logger.log(
                f'could not understand kind: "{kind}". Please choose from "summary", "db", or "campaigns"',
                "ERROR",
            )
