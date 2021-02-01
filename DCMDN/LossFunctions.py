'''
Created on Sep 4, 2017

@author: Kai Polsterer, Antonio D'Isanto
HITS gGmbH, creative commons 4.0 BY-NC-SA
'''

import math
import theano.tensor as T

def phi(x):
    return 1.0 / math.sqrt(2.0 * math.pi) * T.exp(- T.square(x) / 2.0)

def Phi(x):
    return 0.5 * (1.0 + T.erf(x/math.sqrt(2.0)))

def A(mu, var):
    return 2.0*T.sqrt(var) * phi(mu/T.sqrt(var)) + mu * (2.0 * Phi(mu/T.sqrt(var)) - 1)

def cdf(mu, sigma, x):
    return 0.5 * (1.0 + T.erf( (x-mu) / (sigma * math.sqrt(2.0))))

def pdf(mu, sigma, x):
    return 1.0 / T.sqrt(2.0 * math.pi*T.square(sigma)) * T.exp(-T.square(x-mu)/(2.0 * T.square(sigma)))

def GaussianMixtureCRPS(outputs, numberOfGaussians, y):
    means, sigmas, scalers = decomposeGaussianComponents(outputs, numberOfGaussians)
    covars = T.sqr(sigmas)
    crps = scalers[:,0].flatten() * A(y.flatten()-means[:,0].flatten(), covars[:,0].flatten())
    for m in range(1, numberOfGaussians):
        crps = crps + scalers[:,m].flatten() * A(y.flatten()-means[:,m].flatten(), covars[:,m].flatten())
    for m in range(numberOfGaussians):
        for n in range(numberOfGaussians):
            crps = crps - 0.5 * scalers[:,m].flatten() * scalers[:,n].flatten() * A(means[:,m].flatten() - means[:,n].flatten(), covars[:,m].flatten() + covars[:,n].flatten())
    return T.mean(crps)

def GaussianMixtureCDF(outputs, numberOfGaussians, y):
    means, sigmas, scalers = decomposeGaussianComponents(outputs, numberOfGaussians)
    pitValue = (scalers[:,0] * cdf(means[:,0], sigmas[:,0], y)).flatten()
    for t in range(1, numberOfGaussians):
        pitValue = pitValue + (scalers[:,t] * cdf(means[:,t], sigmas[:,t], y)).flatten()
    return pitValue

def decomposeGaussianComponents(outputs, numberOfGaussians):
    means = outputs[:,0:numberOfGaussians]
    sigmas = T.exp(outputs[:,numberOfGaussians:numberOfGaussians*2])
    scalers = T.exp(outputs[:,numberOfGaussians*2:])
    scalers = scalers / T.sum(scalers,axis=1).flatten()[:,None]
    return means, sigmas, scalers

def negativeLogLikelyhood(outputs, numberOfGaussians, y):
    means, sigmas, scalers = decomposeGaussianComponents(outputs, numberOfGaussians)
    variance = T.square(sigmas)
    likelyhoods =  T.log(T.transpose(scalers)) + T.log((1.0/(2.0*math.pi*T.transpose(variance)))) - T.square(y - T.transpose(means)) / (2.0*T.transpose(variance))
    logLikelyhood = -(T.log(T.sum(T.exp(likelyhoods-T.max(likelyhoods,axis=0)), axis=0).flatten()) + T.max(likelyhoods,axis=0))
    return T.mean(logLikelyhood)
