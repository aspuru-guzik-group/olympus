#!/usr/bin/env python

import os
import shutil
import argparse
import subprocess

# ===============================================================================

from olympus import __home__, Logger

# ===============================================================================


class ParserUpload:
    def __init__(self, subparsers):

        self.parser = subparsers.add_parser("upload", help=">> help for upload")
        self.parser.add_argument("-n", "--name", required=True)
        self.parser.add_argument("-p", "--path", default="./")
        self.parser.add_argument(
            "--no-fork", dest="fork", action="store_false",
        )
        self.parser.set_defaults(fork=True)

    @staticmethod
    def __call__(args):
        name = args.name
        path = args.path
        fork = args.fork

        # run some checks to make sure that the dataset is ok
        complete = True
        expected_files = ["dataset.zip", "data.csv", "description.txt"]
        for expected_file in expected_files:
            if not os.path.isfile(f"{path}/{expected_file}"):
                complete = False
                Logger.log(
                    f"could not find expected file {path}/{expected_file}", "ERROR"
                )
        if complete is False:
            Logger.log(
                "Please provide the dataset in the expected format and try again.",
                "INFO",
            )
            return

        # check that we have git
        git_path = shutil.which("git")
        if len(git_path) == 0:
            Logger.log(
                "Could not find a local version of git. Please install git or submit the pull request manually",
                "ERROR",
            )
            return

        # check that we have git-pull-request
        try:
            import git_pull_request
        except ModuleNotFoundError:
            Logger.log("Could not find a local version of git-pull-request", "ERROR")
            return

        # Inspired from the link below, we run a couple of git commands
        # https://medium.com/mergify/managing-your-github-pull-request-from-the-command-line-89cb6af0a7fa
        Logger.log("uploading dataset", "INFO")

        with open(f"{__home__}/cli/template_push_to_github.sh", "r") as content:
            template = content.read()

        replace_dict = {
            "{@DATASET_NAME}": name,
            "{@PATH}": f"../{path}",
            "{@NO_FORK}": "" if fork is True else "--no-fork",
        }

        for key, value in replace_dict.items():
            template = template.replace(str(key), str(value))

        push_file_name = "./push_to_github.sh"
        with open(push_file_name, "w") as content:
            content.write(template)

        # get default git editor
        editor_file = ".editor"
        subprocess.call(f"git config --get core.editor > {editor_file}", shell=True)
        with open(editor_file, "r") as content:
            editor = content.read().strip()
        subprocess.call(
            f"bash push_to_github.sh < {__home__}/cli/template_sign_message_{editor}.sh",
            shell=True,
        )

        # remove push script
        os.remove(push_file_name)


# ===============================================================================
