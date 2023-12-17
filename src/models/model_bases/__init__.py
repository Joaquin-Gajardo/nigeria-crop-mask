r"""
All the model bases, which take as input the raw timeseries
and return a vector
"""
from .rnn import LSTM, GRU


STR2BASE = {"lstm": LSTM, "gru": GRU}


__all__ = ["STR2BASE"]
