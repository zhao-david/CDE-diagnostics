'''
Created on Sep 7, 2017

@author: Kai Polsterer, Antonio D'Isanto
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import numpy
import theano.tensor as T
import ConvolutionalLayer
import ConvolutionalPoolingLayer
import DeepConvolutionalMixtureDensityNetwork
import MixtureDensityNetwork
import WeightInitialization
import ImageAccess
import LossFunctions
import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot

if __name__ == "__main__":
    randomNumberGenerator = numpy.random.RandomState(23456)
    numberOfGaussians = 5 

    x = T.ftensor4("x") # the input data are images
    y = T.fvector("y") # the output data is a 1D vector of labels
    
    nkerns=[128, 256, 512, 1024]
    batchSize = 1000
    layers = []

    if False: # architecture 1 as used for quasars in the publication
        nkerns=[256, 512, 1024]
        filterShape = (nkerns[0], 15, 3, 3)
        layers.append(ConvolutionalPoolingLayer(
                inputs=x.reshape((batchSize, 15, 16, 16)),
                inputShape=(batchSize, 15, 16, 16),
                filterShape=filterShape,
                poolShape=(2,2),
                ignoreBorder=True,
                randomNumberGenerator=randomNumberGenerator,
                weights=WeightInitialization.orthogonalInitialization(shape=filterShape, randomNumberGenerator=randomNumberGenerator),
                activation=T.tanh
        ))
      
        filterShape = (nkerns[1], nkerns[0], 2, 2)
        layers.append(ConvolutionalPoolingLayer(
                inputs=layers[-1].getOutputs(),
                inputShape=(batchSize, nkerns[0], 7, 7),
                filterShape=filterShape,
                poolShape=(2,2),
                ignoreBorder=True,
                randomNumberGenerator=randomNumberGenerator,
                weights=WeightInitialization.orthogonalInitialization(shape=filterShape, randomNumberGenerator=randomNumberGenerator),
                activation=T.tanh
        ))
      
        filterShape = (nkerns[2], nkerns[1], 2, 2)
        layers.append(ConvolutionalLayer.ConvolutionalLayer(
                inputs=layers[-1].getOutputs(),
                inputShape=(batchSize, nkerns[1], 3, 3),
                filterShape=filterShape,
                randomNumberGenerator=randomNumberGenerator,
                weights=WeightInitialization.orthogonalInitialization(shape=filterShape, randomNumberGenerator=randomNumberGenerator),
                activation=T.nnet.relu
        ))
     
        layers.append(MixtureDensityNetwork.MixtureDensityNetwork(
                inputs=layers[-1].getOutputs().flatten(2),
                architecture=[nkerns[2] * 2 * 2, 500, 100, numberOfGaussians*3],
                randomNumberGenerator=randomNumberGenerator,
                activation=T.tanh
        ))
    
    if True: #best performing in the quasar experiment 
        nkerns=[128, 256, 512, 1024]
        filterShape = (nkerns[0], 15, 3, 3)
        layers.append(ConvolutionalLayer.ConvolutionalLayer(
                inputs=x.reshape((batchSize, 15, 16, 16)),
                inputShape=(batchSize, 15, 16, 16),
                filterShape=filterShape,
                randomNumberGenerator=randomNumberGenerator,
                weights=WeightInitialization.orthogonalInitialization(shape=filterShape, randomNumberGenerator=randomNumberGenerator),
                activation=T.tanh
        ))
     
        filterShape = (nkerns[1], nkerns[0], 3, 3)
        layers.append(ConvolutionalLayer.ConvolutionalLayer(
                inputs=layers[-1].getOutputs(),
                inputShape=(batchSize, nkerns[0], 14, 14),
                filterShape=filterShape,
                randomNumberGenerator=randomNumberGenerator,
                weights=WeightInitialization.orthogonalInitialization(shape=filterShape, randomNumberGenerator=randomNumberGenerator),
                activation=T.tanh
        ))
        
        filterShape = (nkerns[2], nkerns[1], 3, 3)
        layers.append(ConvolutionalLayer.ConvolutionalLayer(
                inputs=layers[-1].getOutputs(),
                inputShape=(batchSize, nkerns[1], 12, 12),
                filterShape=filterShape,
                randomNumberGenerator=randomNumberGenerator,
                weights=WeightInitialization.orthogonalInitialization(shape=filterShape, randomNumberGenerator=randomNumberGenerator),
                activation=T.tanh
        ))
     
        filterShape = (nkerns[3], nkerns[2], 3, 3)
        layers.append(ConvolutionalLayer.ConvolutionalLayer(
                inputs=layers[-1].getOutputs(),
                inputShape=(batchSize, nkerns[2], 10, 10),
                filterShape=filterShape,
                randomNumberGenerator=randomNumberGenerator,
                weights=WeightInitialization.orthogonalInitialization(shape=filterShape, randomNumberGenerator=randomNumberGenerator),
                activation=T.nnet.relu
        ))
    
        layers.append(MixtureDensityNetwork.MixtureDensityNetwork(
                inputs=layers[-1].getOutputs().flatten(2),
                architecture=[nkerns[3] * 8 * 8, 500, 100, numberOfGaussians*3],
                randomNumberGenerator=randomNumberGenerator,
                activation=T.tanh
        ))

    data = ImageAccess.generateDataFromFile("SDSS_quasar_catalog_all_coord.csv", "compressedImageCatalog.npz", splits=[100000, 120000, 185000], limits=[6,22])
    
#    costFunction = LossFunctions.GaussianMixtureCRPS(layers[-1].getOutputs(), numberOfGaussians, y) # calculate the CRPS loss based on the last layer
    costFunction = LossFunctions.negativeLogLikelyhood(layers[-1].getOutputs(), numberOfGaussians, y) # calculate the log likelihood loss based on the last layer
    dcmdn = DeepConvolutionalMixtureDensityNetwork.DeepConvolutionalMixtureDensityNetwork(layers, numberOfGaussians, x, y, randomNumberGenerator=randomNumberGenerator)
#     dcmdn.trainConvNet(data[:4], learningRate=0.01, batchSize=batchSize, nEpochs=1000, interval=5, autosave="modelB_", loadfile="modelB_5_0.157234_0.158484.npy", loss="CRPS")
    dcmdn.trainConvNet(data[:4], learningRate=0.01, batchSize=batchSize, nEpochs=1000, interval=5, autosave=None, loadfile=None, costFunction=costFunction)
#     dcmdn.load("modelB_5_0.157234_0.158484.npy")
    predictions = dcmdn.predictValues(data[4], batchSize=batchSize).reshape(-1, numberOfGaussians*3)

    trueRedshifts = data[5].eval()
    CRPS = numpy.asarray(LossFunctions.GaussianMixtureCRPS(predictions, numberOfGaussians=numberOfGaussians, y=trueRedshifts).eval())
#     logLikelyhood = numpy.asarray(negativeLogLikelyhood(predictions, numberOfGaussians=numberOfGaussians, y=trueRedshifts).eval())
    CDF = numpy.asarray(LossFunctions.GaussianMixtureCDF(predictions, numberOfGaussians=numberOfGaussians, y=trueRedshifts).eval())
    mus, sigmas, weights = LossFunctions.decomposeGaussianComponents(predictions, numberOfGaussians)
    sigmas = sigmas.eval()
    weights = weights.eval()
    print (CRPS)#, logLikelyhood)

    pyplot.figure()
    pyplot.hist(CDF,bins=30)
    pyplot.savefig("pit.png")