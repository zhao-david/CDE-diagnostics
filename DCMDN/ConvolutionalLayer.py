'''
Created on Aug 29, 2017

@author: Kai Polsterer, Antonio D'Isanto
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d

class ConvolutionalLayer(object):
    """a very simple and plain convolutional layer of a convolutional network """

    def __init__(self, inputs, inputShape, filterShape, randomNumberGenerator=numpy.random.RandomState(), weights=None, biases=None, activation=T.tanh):
        assert inputShape[1] == filterShape[1]
        if weights is None: # if no weights are specified use zeros
            weights = numpy.zeros(filterShape)
        assert weights.shape == filterShape
        if biases is None: # the biases are the offsets for the individual feature maps and initialized with 0 if not specified else
            biases = numpy.zeros(filterShape[0])
        self.__inputShape = inputShape
        self.__filterShape = filterShape
        self.__randomNumberGenerator = randomNumberGenerator
        self.__weights = theano.shared(value=numpy.asarray(weights, dtype=theano.config.floatX), borrow=True)
        self._biases = theano.shared(value=numpy.asarray(biases, dtype=theano.config.floatX), borrow=True)
        # convolve inputs with filters to generate output
        self._convolutionResult = conv2d(
            input = inputs,
            filters = self.__weights,
            filter_shape = filterShape,
            input_shape = inputShape
        )
        # add biases to the convolved result and apply the specified activation function
        self._outputs = activation(self._convolutionResult + self._biases.dimshuffle('x', 0, 'x', 'x'))

    def getParameters(self):
        return [self.__weights, self._biases]

    def getOutputs(self):
        return self._outputs

    def getWeights(self):
        return self.__weights

    def setWeights(self, weights):
        self.__weights = weights

    def getBiases(self):
        return self._biases

    def setBiases(self, biases):
        self._biases = biases