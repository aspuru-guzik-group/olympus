import olympus


def test_name():
    try:
        assert olympus.__name__ == "olympus"
    except Exception as e:
        raise e
