import sampling
import benchmarks
import numpy as np

problem = benchmarks.mw1()

samples = sampling.random(problem, 15)

print(samples['X'])
print(samples['F'])
print(samples['G'])
