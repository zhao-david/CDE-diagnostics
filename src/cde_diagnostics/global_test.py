import numpy as np
import pandas as pd
from .classifiers import classifier_dict


def global_test(x_train, pit_train, x_test, alphas=np.linspace(0.0, 1.0, 11), clf_name='MLP', n_trials=1000,
                return_T=False):
    
    clf = classifier_dict[clf_name]
    
    ### calculate T_i value on all test points
    all_rhat_alphas = {}
    for alpha in alphas:
        ind_train = [1*(x<=alpha) for x in pit_train]
        rhat = clf
        rhat.fit(X=x_train, y=ind_train)
        all_rhat_alphas[alpha] = rhat.predict_proba(x_test)[:, 1]
    all_rhat_alphas = pd.DataFrame(all_rhat_alphas)
    Ti_values = ((all_rhat_alphas - alphas)**2).sum(axis=1) / len(alphas)
    
    ### refit the classifier using Unif[0,1] random values in place of true PIT values
    all_unif_Ti_values = {}
    for k in range(n_trials):
        Ti_values_k = {}
        all_rhat_alphas_k = {}
        unif_values = np.random.uniform(size=pit_train.shape[0])
        for alpha in alphas:
            ind_values_k = [1*(x<=alpha) for x in unif_values]
            rhat_k = clf
            rhat_k.fit(X=x_train, y=ind_values_k)
            all_rhat_alphas_k[alpha] = rhat_k.predict_proba(x_test)[:, 1]
        all_rhat_alphas_k = pd.DataFrame(all_rhat_alphas_k)
        Ti_values_k = ((all_rhat_alphas_k - alphas)**2).sum(axis=1) / len(alphas)
        all_unif_Ti_values[k] = Ti_values_k
    
    ### compute global p-value
    global_pvalue = sum(1 * (Ti_values.sum() < pd.DataFrame(all_unif_Ti_values).sum(axis=0))) / len(all_unif_Ti_values)
    
    if return_T:
        return global_pvalue, Ti_values.sum()
    else:
        return global_pvalue
