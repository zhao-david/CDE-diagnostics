'''
Created on Sep 4, 2017

@author: Kai Polsterer, Antonio D'Isanto
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import timeit
import numpy
import theano
import theano.tensor as T
from LossFunctions import negativeLogLikelyhood

class DeepConvolutionalMixtureDensityNetwork(object):

    def __init__(self, layers, numberOfGaussians, x, y, randomNumberGenerator=numpy.random.RandomState()):
        self.__layers = layers
        self.__numberOfGaussians = numberOfGaussians
        self.__x = x
        self.__y = y
        self.__randomNumberGenerator = randomNumberGenerator
        # collect all used parameter for the gradient descent
        self.__params = self.__layers[-1].getParameters()
        for i in range(len(self.__layers)-1):
            self.__params += self.__layers[i].getParameters()

    def prepareNetworkArchitecture(self, learningRate=0.01, batchSize=100, costFunction=None):
        print("defining the cost functions ...")
        index = T.iscalar()
        if costFunction is None:
            costFunction = negativeLogLikelyhood(self.__layers[-1].getOutputs(), self.__numberOfGaussians, self.__y)

        grads = T.grad(costFunction, self.__params)

        updates = [
            (param_i, (param_i - learningRate * grad_i)*numpy.float32(self.__randomNumberGenerator.binomial(1,0.6,(param_i.eval().shape)))) # the binomial function adds dropout, with a 60% of weights switched off
            for param_i, grad_i in zip(self.__params, grads)
        ]

        print("defining the training function ...")
        self.__trainModel = theano.function(
            [index],
            costFunction,
            updates=updates,
            givens={
                self.__x: self.__trainX[index * batchSize: (index + 1) * batchSize],
                self.__y: self.__trainY[index * batchSize: (index + 1) * batchSize]
            }
        )

        print("defining the validation function ...")
        self.__validateModel = theano.function(
            [index],
            costFunction,
            givens={
                self.__x: self.__validX[index * batchSize: (index + 1) * batchSize],
                self.__y: self.__validY[index * batchSize: (index + 1) * batchSize]
            }
        )

    def save(self, filename):
        print("saving the network ...")
        parameters = []
        for i in range(len(self.__params)):
            parameters.append(self.__params[i].get_value())
        numpy.save(filename, (parameters))

    def load(self, filename):
        print("loading the network ...")
        parameters = numpy.load(filename)
        for i in range(len(self.__params)):
            self.__params[i].set_value(parameters[i])

    def getParameters(self):
        return self.__params

    def trainConvNet(self, inputData, learningRate=0.01, batchSize=100, nEpochs=500, interval=5, autosave=None, loadfile=None, costFunction=None):
        print("training the network ...")
        self.__trainX, self.__trainY, self.__validX, self.__validY = inputData

        # compute number of minibatches for training, validation and testing
        nTrainBatches = self.__trainX.get_value(borrow=True).shape[0] // batchSize
        nValidBatches = self.__validX.get_value(borrow=True).shape[0] // batchSize

        self.prepareNetworkArchitecture(learningRate=learningRate, batchSize=batchSize, costFunction=costFunction)

        if loadfile != None:
            self.load(loadfile)

        startTime = timeit.default_timer()
        epoch = 0
        done_looping = False
        while (epoch < nEpochs) and (not done_looping):
            epoch = epoch + 1
            trainingLosses = []
            for minibatchIndex in range(nTrainBatches):
                trainingLosses.append(self.__trainModel(minibatchIndex))
                if minibatchIndex % 10 == 0:
                    print("epoch %i, batch %i, training error = %f" % (epoch, minibatchIndex, trainingLosses[-1]))

            if epoch % interval == 0:
                validationLosses = [self.__validateModel(i) for i in range(nValidBatches)]
                print("epoch %i, training error %f, validation error %f" % (epoch, numpy.mean(trainingLosses), numpy.mean(validationLosses)))

            if epoch % interval == 0:
                if autosave != None:
                    self.save(autosave+"%i_%f_%f.npy" %(epoch, numpy.mean(trainingLosses), numpy.mean(validationLosses)))

        endTime = timeit.default_timer()
        print("training ended")
        print("the training took %.2f minutes" % ((endTime - startTime) / 60.))

    def predictValues(self, inputData, batchSize=100):
        index = T.iscalar()
        testX = inputData
        predicModel = theano.function(
            [index],
            self.__layers[-1].getOutputs(),
            givens={
                self.__x: testX[index * batchSize: (index + 1) * batchSize]
            }
        )
        nTestBatches = testX.get_value(borrow=True).shape[0] // batchSize
        predictedValues = [predicModel(i) for i in range(nTestBatches)]
        return numpy.asarray(predictedValues)
