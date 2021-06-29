import numpy as np
import pandas as pd
from scipy.stats import binom
from matplotlib import pyplot as plt
from .classifiers import classifier_dict


def local_histogram(x_train, pit_train, x_test, alphas=np.linspace(0.0, 0.999, 101), clf_name='MLP',
                    ci_level=0.05, n_bins=7, figsize=(5,4)):
    
    clf = classifier_dict[clf_name]
    
    ### calculate PIT values at point of interest x_test
    all_rhat_alphas = {}
    for alpha in alphas:
        ind_train = [1*(x<=alpha) for x in pit_train]
        rhat = clf
        rhat.fit(X=x_train, y=ind_train)
        all_rhat_alphas[alpha] = rhat.predict_proba(x_test)[:, 1][0]
    all_rhat_alphas = pd.Series(all_rhat_alphas)
    
    ### create figure
    n = len(all_rhat_alphas)
    fig = all_rhat_alphas.hist(bins=n_bins, figsize=figsize)
    
    ### binomial confidence bands
    low_lim = binom.ppf(q=ci_level/2, n=n, p=1/n_bins)
    upp_lim = binom.ppf(q=1 - ci_level/2, n=n, p=1/n_bins)
    
    plt.axhline(y=low_lim, color='grey')
    plt.axhline(y=upp_lim, color='grey')
    plt.axhline(y=n/n_bins, label='Uniform Average', color='red')
    plt.fill_between(x=np.linspace(0, 1, 100),
                     y1=np.repeat(low_lim, 100),
                     y2=np.repeat(upp_lim, 100),
                     color='grey', alpha=0.2)
    
    plt.title('Estimated local PIT distribution at %s' % str(x_test), fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.close()
    
    return fig

