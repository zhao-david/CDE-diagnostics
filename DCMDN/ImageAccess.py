'''
Created on Aug 29, 2017

@author: Kai Polsterer, Antonio D'Isanto
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import numpy
import theano
import theano.tensor as T

def buildColors(data, imageEdgeSize=28):
    resultX = []
    for i in range(len(data)):
        xValue = data[i].tolist()
        for x in range(len(data[0]) - 1):
            for y in range(x + 1,len(data[0])):
                xValue.append(data[i][x]-data[i][y])
        resultX.append(numpy.array(xValue).reshape((15,imageEdgeSize,imageEdgeSize)))
    return numpy.array(resultX)

def buildRotatedImages(data, z, imageEdgeSize=28):
    print("rotating images ...")
    rotatedImages = []
    redshifts = []
    for k in range(4):
        rotatedImages.append(numpy.rot90(data, k=k, axes=(2,3)))
        redshifts.append(z)
    return numpy.asarray(rotatedImages).reshape(-1,15,imageEdgeSize,imageEdgeSize), numpy.asarray(redshifts).flatten()

def generateDataFromFile(catalogName, compressedImages, splits = [1,2,3], limits = [1,2]):
    catalog = numpy.loadtxt(catalogName, delimiter=',', skiprows=1)
    z = catalog[:,2]
    data = numpy.load(compressedImages) # use a compressed images file to save memory and speed up the loading
    images = []
    key = data.keys()[0]
    images.append([data[key]])
    images = numpy.asarray(images).reshape((len(z),5,28,28))
    print("data loaded ...")

    images = images[:,:,limits[0]:limits[1],limits[0]:limits[1]]
    z = numpy.asarray(z)

    # cleaning of the input images, removing those who contains NaNs, zeros or negative values
    toDelete = []
    for i in range(len(z)):
        if numpy.isnan(images[i]).any() or images[i].any()==0 or images[i].any()<0:
            toDelete.append(i)
            print("image %i deleted" % i)

    images = numpy.delete(images, toDelete, axis=0)
    z = numpy.delete(z, toDelete, axis=0)

    imageEdgeSize = limits[1]-limits[0]
    # build color images
    images = buildColors(images, imageEdgeSize=imageEdgeSize)

    data = list(zip(images, z))

    # random shuffling of the images with the associated redshift
    numpy.random.seed(35325)
    numpy.random.shuffle(data)
    images[:], z[:] = zip(*data)

    # defining training, test and validation sets; apply the desired values
    trainX = images[:splits[0]]
    validX = images[splits[0]:splits[1]]
    testX = images[splits[1]:splits[2]]
    trainY = numpy.array(z[:splits[0]])
    validY = numpy.array(z[splits[0]:splits[1]])
    testY = numpy.array(z[splits[1]:splits[2]])

    # apply data augmentation through image rotation
#     trainX, trainY = buildRotatedImages(trainX, trainY, imageEdgeSize=imageEdgeSize)

    # defines theano variables for the various sets
    trainX = T._shared(numpy.asarray(trainX, dtype=theano.config.floatX), borrow=True)
    trainY = T._shared(numpy.asarray(trainY, dtype=theano.config.floatX), borrow=True)
    validX = T._shared(numpy.asarray(validX, dtype=theano.config.floatX), borrow=True)
    validY = T._shared(numpy.asarray(validY, dtype=theano.config.floatX), borrow=True)
    testX = T._shared(numpy.asarray(testX, dtype=theano.config.floatX), borrow=True)
    testY = T._shared(numpy.asarray(testY, dtype=theano.config.floatX), borrow=True)

    return trainX, trainY, validX, validY, testX, testY
