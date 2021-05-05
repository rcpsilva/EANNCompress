###################################################
# Wrapper for multiobjective optimization problems
###################################################

import numpy as np
from pymoo.factory import get_problem

def mw1():
    return get_problem("mw1")

def mw2():
    return get_problem("mw2")

def mw3():
    return get_problem("mw3")

def mw11():
    return get_problem("mw11")

def mw14():
    return get_problem("mw14")

def dascmop6():
    return get_problem("DASCMOP6",10)
