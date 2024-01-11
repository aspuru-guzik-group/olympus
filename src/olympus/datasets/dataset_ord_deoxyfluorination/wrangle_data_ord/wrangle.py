# @Author: Qianxiang Ai (qai@mit.edu)
import json

import numpy as np
import requests
from ord_schema.message_helpers import load_message, messages_to_dataframe
from ord_schema.proto import dataset_pb2
from pandas._typing import FilePath

"""
download a reaction dataset from the Open Reaction Database and convert it to a benchmark dataset for reaction yield
you can browse ord datasets at https://open-reaction-database.org/browse

additional dependencies:
pip install ord_schema
"""


def download_ord(dataset_url: str, save_as: FilePath):
    r = requests.get(dataset_url)
    with open(save_as, "wb") as f:
        f.write(r.content)


def convert_ord(ord_dataset_file: FilePath):
    dataset = load_message(ord_dataset_file, dataset_pb2.Dataset)
    df = messages_to_dataframe(dataset.reactions)
    to_drop = []
    to_rename = {}
    for c in df.columns:
        if len(set(df[c].tolist())) == 1:
            to_drop.append(c)
        elif "vendor" in c:
            to_drop.append(c)
        elif "provenance" in c:
            to_drop.append(c)
        elif "preparations" in c:
            to_drop.append(c)
        elif "reaction_id" == c:
            to_drop.append(c)
        to_rename[c] = c.replace(" ", "_")
    df.drop(columns=to_drop, inplace=True)
    df.rename(columns=to_rename, inplace=True)
    columns = df.columns.tolist()
    continuous_columns = df.select_dtypes(include=np.number).columns.tolist()
    targets = [c for c in continuous_columns if "outcome" in c]  # better double check
    target = [t for t in targets if "percentage.value" in t][0]  # error out if no percentage yield reported
    params = [c for c in columns if c not in targets and "outcome" not in c]

    continuous_variables = [c for c in params if c in continuous_columns]
    categorical_variables = [c for c in params if c not in continuous_variables]

    config_params = []
    for continuous_variable in continuous_variables:
        param = {"name": continuous_variable, "type": "continuous", "low": df[continuous_variable].min(),
                 "high": df[continuous_variable].max()}
        config_params.append(param)

    for categorical_variable in categorical_variables:
        param = {"name": categorical_variable, "type": "categorical",
                 "options": sorted(set(df[categorical_variable].tolist()))}
        config_params.append(param)

    config_dict = {
        "constraints": {
            "parameters": "none",
            "measurements": "positive",
            "known": "no"
        },
        "parameters": config_params,
        "measurements": [
            {"name": target, "type": "continuous"}
        ],
        "default_goal": "maximize"
    }

    summary_features = [f"\t\t{feat}\tcontinuous" for feat in continuous_variables]
    summary_features += [f"\t\t{feat}\tcategorical" for feat in categorical_variables]
    summary_features = "\n".join(summary_features)
    summary = f"""
=========================================
                Summary
-----------------------------------------
    Number of Samples             {len(df)}
    Dimensionality                 {len(df.columns) - 1}
    Features:
{summary_features}
    Targets:
        {target}         continuous
=========================================
"""

    return df, config_dict, summary


def export_dataset(dataset_url: str):
    download_ord(dataset_url, "ord_dataset.pb.gz")
    df, config, summary = convert_ord("ord_dataset.pb.gz")
    df.to_csv("../data.csv", index=False)
    with open("../config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open("../description.txt", "w") as f:
        f.write(f"Exported from {dataset_url}\n{summary}")


if __name__ == '__main__':
    export_dataset(
        dataset_url="https://github.com/open-reaction-database/ord-data/blob/main/data/d2/ord_dataset-d26118acda314269becc35db5c22dc59.pb.gz?raw=true")
