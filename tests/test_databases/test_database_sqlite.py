#!/usr/bin/env python
import pytest

from olympus.campaigns import Campaign
from olympus.databases import Database


def test_creation():
    db = Database(kind="sqlite")


def test_add():
    db = Database(kind="sqlite")
    campaign = Campaign()
    db.log_campaign(campaign)


def test_list_campaign_ids():
    db = Database(kind="sqlite")
    campaign = Campaign()
    db.log_campaign(campaign)
    import time

    time.sleep(0.1)
    campaign_ids = db.list_campaigns()
    assert campaign.id in campaign_ids


if __name__ == "__main__":
    test_add()
    test_list_campaign_ids()
