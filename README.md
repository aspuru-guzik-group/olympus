Need to use Python 3.7 because we require `tensorflow==1.15`

Deploying on Streamlit requires a single `requirements.txt`, but installing `phoenics` from pypa will fail because its `setup.py` imports `numpy`, which gives an ImportError as it is not yet installed at that point. For this reason, the phoenics source has been pulled here and manually modified so that it does not give the ImportError.

Essentially, to install `phoenics`, they want you to do:
```
pip install numpy
pip install phoenics
```

But deploying on Streamlit requires a single command `pip install -r requirements.txt`


`dragonfly-opt` has the same issue.


