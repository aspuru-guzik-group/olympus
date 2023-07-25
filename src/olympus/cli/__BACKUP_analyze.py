#!/usr/bin/env python
import argparse

from olympus.cli.main import check_unknown_cmd


# ======================
# Parse Option Arguments
# ======================
def parse_options():

    parser = argparse.ArgumentParser(description="It does this and that.")

    parser.add_argument(
        "-g",
        metavar="generator",
        dest="generatot",
        type=str,
        help="the generator to use",
        default="gpyopt",
    )
    parser.add_argument(
        "-e",
        metavar="evaluator",
        dest="evaluator",
        type=str,
        help="the evaluator to use",
        default="dummy",
    )

    args, unknown = parser.parse_known_args()
    check_unknown_cmd(unknown)

    return args


# ====
# Main
# ====
def main(args):
    pass


def entry_point():
    args = parse_options()
    main(args)


if __name__ == "__main__":
    entry_point()
