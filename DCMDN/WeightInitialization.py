'''
Created on Aug 29, 2017

@author: Kai Polsterer
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import numpy
import math

def orthogonalInitialization(shape, randomNumberGenerator=numpy.random.RandomState()):
    # initialize with random values with an orthogonal initialization as suggested in http://arxiv.org/abs/1312.6120
    randomValues = randomNumberGenerator.random_sample((shape[0], numpy.prod(shape[1:])))
    u, _sigma, v = numpy.linalg.svd(randomValues, full_matrices=False)
    if u.shape == randomValues.shape:
        return u.reshape(shape)
    else:
        return v.reshape(shape)

def LeCunInitialization(shape, randomNumberGenerator=numpy.random.RandomState()):
    # initialize with a uniform distribution http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    indegree = numpy.prod(shape[1:])
    outdegree = shape[0]
    return math.sqrt(6.0) * randomNumberGenerator.uniform(low = -1, high = 1, size=shape) / math.sqrt(indegree + outdegree)