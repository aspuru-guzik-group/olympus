#!/usr/bin/env python

import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("-f", dest="file", type=str, help="Pickle file")
parser.add_argument("-k", dest="kind", help="New dataset kind/name", type=str)
args = parser.parse_args()

# load
with open(args.file, "rb") as content:
    emulator = pickle.load(content)

# rename
emulator.dataset.kind = args.kind

# save
with open("new_emulator.pickle", "wb") as content:
    emulator = pickle.dump(emulator, content)
