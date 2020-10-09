#!/bin/bash

# clone repository
git clone https://github.com/FlorianHase/olympus_datasets.git
# navigate into repository
cd olympus_datasets
# find branch "datasets"
git checkout datasets
# set editor
git config core.editor vi
# create new dataset branch
git branch new-dataset-{@DATASET_NAME}
# switch to new dataset branch
git checkout new-dataset-{@DATASET_NAME}
# create directory
mkdir ./src/olympus_datasets/datasets/dataset_{@DATASET_NAME}
# move dataset files to expected location
cp -r {@PATH}/* ./src/olympus_datasets/datasets/dataset_{@DATASET_NAME}
# add files
git add .
# commit
git commit -a -m "added dataset {@DATASET_NAME}"
# submit pull-request
git pull-request --target-branch datasets {@NO_FORK} -m "added dataset {@DATASET_NAME}"
# remove repository
cd ../
rm -rf ./olympus_datasets
