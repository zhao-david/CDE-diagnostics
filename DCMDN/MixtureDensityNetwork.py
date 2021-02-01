'''
Created on Aug 29, 2017

@author: Kai Polsterer, Antonio D'Isanto
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import numpy
import theano
import theano.tensor as T

class MixtureDensityNetwork(object):

    def __init__(self, inputs, architecture=[2,1], randomNumberGenerator=numpy.random.RandomState(), activation=None, createArchitecture=True):
        assert len(architecture) >= 2, "minimum of 2 layers required"
        self.__inputs = inputs
        self.__architecture = architecture
        self.__activation = activation
        self.__randomNumberGenerator = randomNumberGenerator
        if createArchitecture:
            self.createNetworkArchitecture()

    def calculateWeights(self, inNodes, outNodes):
        if self.__activation == T.tanh or self.__activation is None:
            return numpy.asarray(
                self.__randomNumberGenerator.uniform(
                    low=-numpy.sqrt(6. / (inNodes + outNodes)),
                    high=numpy.sqrt(6. / (inNodes + outNodes)),
                    size=(inNodes, outNodes)
                ),
                dtype=theano.config.floatX)
        elif self.__activation == T.nnet.sigmoid:
            return 4*numpy.asarray(
                self.__randomNumberGenerator.uniform(
                    low=-numpy.sqrt(6. / (inNodes + outNodes)),
                    high=numpy.sqrt(6. / (inNodes + outNodes)),
                    size=(inNodes, outNodes)
                ),
                dtype=theano.config.floatX)

    def createNetworkArchitecture(self):
        inputToLayer = self.__inputs
        params = []
        self.__weights = []
        self.__biases = []
        for i in range(len(self.__architecture) - 1):
            weights = theano.shared(value=self.calculateWeights(self.__architecture[i],self.__architecture[i+1]), name='weights L'+str(i), borrow=True)
            biases = theano.shared(value=numpy.zeros((self.__architecture[i+1],), dtype=theano.config.floatX), name='biases L'+str(i), borrow=True)
            self.__weights.append(weights)
            self.__biases.append(biases)
            inputToLayer = T.dot(inputToLayer, weights) + biases
            if ((self.__activation != None) and (i + 1 < len(self.__architecture) - 1)):
                inputToLayer = self.__activation(inputToLayer)
            params = [weights, biases] + params
        self.__outputs = inputToLayer
        self.__params = params

    def getParameters(self):
        return self.__params

    def getOutputs(self):
        return self.__outputs

    def getWeights(self):
        return self.__weights

    def setWeights(self, weights):
        self.__weights = weights

    def getBiases(self):
        return self.__biases

    def setBiases(self, biases):
        self.__biases = biases
