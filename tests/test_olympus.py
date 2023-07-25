#!/usr/bin/env python

import pytest

from olympus import Olympus
from olympus.campaigns import Campaign
from olympus.planners import planner_names


@pytest.mark.parametrize("planner_kind", planner_names)
def test_olympus_run(planner_kind):
    olymp = Olympus()
    olymp.run(campaign=Campaign(), planner=planner_kind)


if __name__ == "__main__":
    test_olympus_run("BasinHopping")
