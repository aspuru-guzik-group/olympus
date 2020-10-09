#!/usr/bin/env python

import os
import subprocess

from olympus import __home__
from olympus import Logger


class ConnectorGithub:

    URL = "https://github.com/FlorianHase/olympus_datasets/branches/datasets/src/olympus_datasets/"

    def __init__(self):
        pass

    def list(self):
        Logger.log("connecting to github", "INFO")
        tmp_file = "remote_folders"
        remote_datasets = []
        subprocess.call(f"svn ls -R {self.URL} > {tmp_file}", shell=True)
        with open(tmp_file, "r") as content:
            for line in content:
                dataset_name = line.split("/")[0]
                if not dataset_name in remote_datasets:
                    remote_datasets.append(dataset_name)
        os.remove(tmp_file)
        remote_datasets = [remote_dataset[8:] for remote_dataset in remote_datasets]
        return sorted(remote_datasets)

    def get_dataset(self, dataset_name):
        url = f"{self.URL}/datasets/dataset_{dataset_name}"
        subprocess.call(
            f"svn export {url} {__home__}/datasets/dataset_{dataset_name}", shell=True
        )
        # check if expected files exist
        expected_files = ["dataset.zip", "data.csv", "description.txt"]
        success = True
        for expected_file in expected_files:
            success = success and os.path.isfile(
                f"{__home__}/datasets/dataset_{dataset_name}/{expected_file}"
            )
        return success

    def get_baseline(self):
        url = f"{self.URL}/baseline/"
        subprocess.call(
            f"svn export {url} {__home__}/baseline/random_baseline", shell=True
        )
        success = True
        return success
