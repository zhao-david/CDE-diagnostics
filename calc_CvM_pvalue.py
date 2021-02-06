import argparse
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.neural_network import MLPClassifier



# calculate p-value at point x_i by resampling (x_i, U_i) where U_i is uniform
def main(pit_values_dict_name, x_test_name, alphas=np.linspace(0.1, 0.9, 9), n_trials=1000):
    
    with open(pit_values_dict_name, 'rb') as handle:
        pit_values_dict = pickle.load(handle)
    
    x_test = np.load(x_test_name)
    
    x_range = np.linspace(-2,2,41)
    x1, x2 = np.meshgrid(x_range, x_range)
    grid = np.hstack([x1.ravel().reshape(-1,1), x2.ravel().reshape(-1,1)])
    
    # calculate T_i values across grid of x_i values
    Ti_values = {}
    all_rhat_alphas = {}
    for name, pit in pit_values_dict.items():
        all_rhat_alphas[name] = {}
        for alpha in alphas:
            ind_values = [1*(x<=alpha) for x in pit]
            rhat = MLPClassifier(alpha=0, max_iter=25000)
            rhat.fit(X=x_test, y=ind_values)

            # fit rhat at each point in prediction grid
            all_rhat_alphas[name][alpha] = rhat.predict_proba(grid)[:, 1]

        # rhat_alphas for all alphas at all points
        all_rhat_alphas[name] = pd.DataFrame(all_rhat_alphas[name])

        # compute Ti summary statistic
        Ti_values[name] = ((all_rhat_alphas[name] - alphas)**2).sum(axis=1) / len(alphas)
    
    date_str = datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    with open('Ti_values_' + date_str + '.pkl', 'wb') as handle:
        pickle.dump(Ti_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('all_rhat_alphas_' + date_str + '.pkl', 'wb') as handle:
        pickle.dump(all_rhat_alphas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # refit the classifier using Unif[0,1] random values in place of true PIT values
    pbar = tqdm(total=n_trials, desc='Simulated Ti for Uniform Values')
    all_unif_Ti_values = {}
    all_rhat_classifiers = {}
    for k in range(n_trials):
        Ti_values_k = {}
        all_rhat_alphas_k = {}
        all_rhat_classifiers[k] = {}
        for name, pit in pit_values_dict.items():
            all_rhat_alphas_k[name] = {}
            unif_values = np.random.uniform(size=pit.shape[0])
            all_rhat_classifiers[k][name] = {}
            for alpha in alphas:
                ind_values_k = [1*(x<=alpha) for x in unif_values]
                rhat_k = MLPClassifier(alpha=0, max_iter=25000)
                rhat_k.fit(X=x_test, y=ind_values_k)

                # fit rhat at each point in prediction grid
                all_rhat_alphas_k[name][alpha] = rhat_k.predict_proba(grid)[:, 1]
                
                # store rhat so we can use it to build confidence bands for plots
                all_rhat_classifiers[k][name][alpha] = rhat_k

            # rhat_alphas for all alphas at all points
            all_rhat_alphas_k[name] = pd.DataFrame(all_rhat_alphas_k[name])

            # compute Ti summary statistic
            Ti_values_k[name] = ((all_rhat_alphas_k[name] - alphas)**2).sum(axis=1) / len(alphas)
        
        all_unif_Ti_values[k] = Ti_values_k
    
        pbar.update(1)
    
    date_str = datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    with open('all_unif_Ti_values_' + date_str + '.pkl', 'wb') as handle:
        pickle.dump(all_unif_Ti_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('all_rhat_classifiers_' + date_str + '.pkl', 'wb') as handle:
        pickle.dump(all_rhat_classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pit_values_dict_name', action="store", type=str, default=None,
                        help='Name of pickled dictionary of PIT values')
    parser.add_argument('--x_test_name', action="store", type=str, default=None,
                        help='Name of saved numpy array of x_test values')
    argument_parsed = parser.parse_args()
    
    main(
        pit_values_dict_name=argument_parsed.pit_values_dict_name,
        x_test_name=argument_parsed.x_test_name
    )
