#!/usr/bin/env python

import os
import subprocess

from olympus import Logger, __home__

# ===============================================================================


# ===============================================================================

try:
    import requests
except ModuleNotFoundError:
    Logger.log("module <requests> required to download dataset", "FATAL")

# ===============================================================================


class ConnectorServer:

    URL = "http://olympus.datasets.ngrok.io"

    def __init__(self):
        pass

    def list(self):
        Logger.log("connecting to server", "INFO")
        url = f"{self.URL}/list_datasets"
        print("URL", url)
        response = requests.post(url, data={})
        if response.status_code == 200:
            datasets = response.json()["datasets"]
            return sorted(datasets)
        else:
            return self._process_response(response)

        return self._process_response(response)

    def get_dataset(self, dataset_name):
        success = self._check_dataset(dataset_name)
        if success:
            Logger.log("downloading dataset", "INFO")
            self._download_dataset(dataset_name)

    def get_baseline(self):
        Logger.log("downloading baseline", "INFO")
        self._download_baseline()

    def _process_response(self, response):
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            Logger.log("could not reach server", "ERROR")
            return False
        else:
            Logger.log("unknown error", "ERROR")
            return False

    def _check_dataset(self, dataset_name):
        Logger.log("connecting to server", "INFO")
        url = f"{self.URL}/check_dataset"
        data = {"dataset_name": dataset_name}
        response = requests.post(url, data=data)
        return self._process_response(response)

    def _download_dataset(self, dataset_name):
        Logger.log(f"downloading dataset {dataset_name}", "INFO")
        url = f"{self.URL}/get_dataset"
        data = {"dataset_name": dataset_name}
        response = requests.post(url, data=data)
        if response.status_code == 200:

            target_dir = f"{__home__}/datasets/dataset_{dataset_name}"
            target_name = f"{target_dir}/dataset.zip"

            try:
                os.makedirs(target_dir)
            except:
                pass

            # save file
            Logger.log("saving dataset", "INFO")
            with open(target_name, "wb") as content:
                content.write(response.content)

            # unzip file
            Logger.log("unpacking dataset", "INFO")
            subprocess.call(f"unzip {target_name} -d {target_dir}", shell=True)
            Logger.log("dataset installed", "INFO")

        elif response.status_code == 204:
            error = response.json()["error"]
            Logger.log(f"did not get dataset ({error})", "ERROR")

    def _download_baseline(self):
        url = f"{self.URL}/get_baseline"
        response = requests.post(url, data={})
        if response.status_code == 200:

            target_dir = f"{__home__}/baseline/"
            target_name = f"{target_dir}/baseline.zip"

            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            Logger.log("saving baseline", "INFO")
            with open(target_name, "wb") as content:
                content.write(response.content)

            # unzip baseline
            Logger.log("unpacking baseline", "INFO")
            subprocess.call(f"unzip {target_name} -d {target_dir}", shell=True)
            Logger.log("baseline installed", "INFO")
            os.remove(target_name)

        elif response.status_code == 204:
            error = response.json()["error"]
            Logger.log(f"did not get baseline ({error})", "ERROR")
