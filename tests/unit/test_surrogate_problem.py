from numpy.core.records import array
import pytest
import numpy as np
from pymoo.model.problem import Problem
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir =os.path.dirname(parentdir)
sys.path.insert(1, grandparentdir) 

import surrogate_problem

