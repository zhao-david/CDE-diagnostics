'''
Created on Aug 29, 2017

@author: Kai Polsterer
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import numpy
import theano.tensor as T
from theano.tensor.signal import pool
from ConvolutionalLayer import ConvolutionalLayer

class ConvolutionalPoolingLayer(ConvolutionalLayer):
    """a pooling Layer on top of a convolutional layer """

    def __init__(self, inputs, inputShape, filterShape, poolShape=(2,2), ignoreBorder=True, randomNumberGenerator=numpy.random.RandomState(), weights=None, biases=None, activation=T.tanh):
        super(ConvolutionalPoolingLayer, self).__init__(inputs, inputShape, filterShape, randomNumberGenerator, weights, biases, activation)
        self._poolingResult = pool.pool_2d(
            input=self._convolutionResult,
            ws=poolShape,
            ignore_border=ignoreBorder
        )
        self._outputs = activation(self._poolingResult + self._biases.dimshuffle('x', 0, 'x', 'x'))