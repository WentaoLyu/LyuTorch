__version__ = "0.0.1"

from .Variables.tape import tape
from ._basic_actions import *
from .tmath import *
from .variable import *
import lyutorch.linalg
import lyutorch.Variables


class no_grad:
    def __enter__(self):
        tape.make_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        tape.make_grad = True
