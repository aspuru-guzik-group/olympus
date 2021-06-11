import numpy as np
from sklearn.metrics import r2_score

np.random.seed(100691)
data1 = np.random.uniform(low=0, high=1, size=(3, 2))
data2 = np.random.uniform(low=0, high=1, size=(3, 2))

def test_r2_score():
    """Test replacement r2_score function."""
    from olympus.utils.misc import r2_score as r2_score0

    assert np.allclose(r2_score(data1, data2), r2_score0(data1, data2))

if __name__ == "__main__":
    test_r2_score()