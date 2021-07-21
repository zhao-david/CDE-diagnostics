import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .classifiers import classifier_dict


def local_pp_plot(x_train, pit_train, x_test, alphas=np.linspace(0.0, 0.999, 101), clf_name='MLP',
                  confidence_bands=False, conf_alpha=0.05, n_trials=100, figsize=(5,4)):
    
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
    fig = all_rhat_alphas.plot(style='.', figsize=(5,4), markersize=7)
    lims = [
        np.min([0,0]),
        np.max([1,1]),
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    
    ### create confidence bands, by refitting the classifier using Unif[0,1] random values in place of true PIT values
    if confidence_bands:
        null_alphas = {}
        for k in range(n_trials):
            Ti_values_k = {}
            all_rhat_alphas_k = {}
            unif_values = np.random.uniform(size=pit_train.shape[0])
            for alpha in alphas:
                ind_values_k = [1*(x<=alpha) for x in unif_values]
                rhat_k = clf
                rhat_k.fit(X=x_train, y=ind_values_k)
                all_rhat_alphas_k[alpha] = rhat_k.predict_proba(x_test)[:, 1][0]
            null_alphas[k] = pd.Series(all_rhat_alphas_k)
            
        lower_band = pd.DataFrame(null_alphas).quantile(q=conf_alpha/2, axis=1)
        upper_band = pd.DataFrame(null_alphas).quantile(q=1-conf_alpha/2, axis=1)
        
        plt.fill_between(alphas,
                         lower_band,
                         upper_band,
                         alpha=0.15
                        )
    
    plt.title("Local coverage of CDE model at %s" % str(x_test), fontsize=20)
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel("$\hat r($" + r'$\alpha$' + "$)$", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.close()
    
    return fig

