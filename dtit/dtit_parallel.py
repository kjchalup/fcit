""" A parallelized conditional independence test.

This implementation uses the joblib library to parallelize test
statistic computation over all available cores. By default, num_perm=8
(instead of num_perm=10 in the non-parallel version) as 8 cores is a
common number on current architectures.

Reference:
Chalupka, Krzysztof and Perona, Pietro and Eberhardt, Frederick, 2017.
"""
import os
import joblib
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import mean_squared_error as mse

def obtain_error(data_and_i):
    """ 
    A function used for multithreaded computation of the dtit test
    statistic (compare with the non-parallel dtit.py implementation).
    data['x']: First variable.
    data['y']: Second variable.
    data['z']: Conditioning variable.
    data['data_permutation']: Permuted indices of the data.
    data['perm_ids']: Permutation for the bootstrap.
    data['n_test']: Number of test points.
    data['clf']: Decision tree regressor.
    data['reshuffle']: Boolean flag. If True, obtain the MSE
        after reshuffling x. If False, obtain the original MSE.
    """
    data, i = data_and_i
    x = data['x']
    y = data['y']
    z = data['z']
    perm_ids = np.random.permutation(x.shape[0])
    data_permutation = data['data_permutation'][i]
    n_test = data['n_test']
    clf = data['clf']

    if z is not None:
        x_z = np.hstack([x, z])
        x_z_bootstrap = np.hstack([x[perm_ids], z])
    else:
        x_z = x
        x_z_bootstrap = x[perm_ids]

    if data['reshuffle']:
        clf.fit(x_z_bootstrap[data_permutation][n_test:],
                y[data_permutation][n_test:])
        return mse(y[data_permutation][:n_test],
            clf.predict(x_z_bootstrap[data_permutation][:n_test]))
    else:
        clf.fit(x_z[data_permutation][n_test:], y[data_permutation][n_test:])
        return mse(y[data_permutation][:n_test],
            clf.predict(x_z[data_permutation][:n_test]))


def test(x, y, z=None, num_perm=8, prop_test=.1,
    discrete=(False, False), plot_return=False, verbose=False,
    max_dim=None, **kwargs):
    """ The neural net probabilistic independence test.

    See Chalupka, Perona, Eberhardt 2017 [arXiv link coming].

    Args:
        x (n_samples, x_dim): First variable.
        y (n_samples, y_dim): Second variable.
        z (n_samples, z_dim): Conditioning variable. If z==None (default),
            then performs an unconditional independence test.
        num_perm: Number of data permutations to estimate
            the p-value from marginal stats.
        prop_test (int): Proportion of data to evaluate test stat on.
        discrete (bool, bool): Whether x or y are discrete.
        plot_return (bool): If True, return statistics useful for plotting.
        verbose (bool): Print out progress messages (or not).
        max_dim (int): If not None, and data.shape[1] > max_dim, use random
            projections to reduce data dimensionality.
        kwargs: Arguments to pass to the neural net constructor.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    # Reduce dimensionality, if desired, using random Gaussian projections.
    if max_dim is not None:
        if x.shape[1] > max_dim:
            x = GaussianRandomProjection(n_components=max_dim).fit_transform(x)
        if y.shape[1] > max_dim:
            y = GaussianRandomProjection(n_components=max_dim).fit_transform(y)
        if z is not None and z.shape[1] > max_dim:
            z = GaussianRandomProjection(n_components=max_dim).fit_transform(z)

    if discrete[0] and not discrete[1]:
        # If x xor y is discrete, use the continuous variable as input.
        x, y = y, x
    elif x.shape[1] < y.shape[1]:
        # Otherwise, predict the variable with fewer dimensions.
        x, y = y, x

    xz_dim = x.shape[1] + (z.shape[1] if z is not None else 0)
    
    #y /= y.flatten().std()
    y = StandardScaler().fit_transform(y)

    # Compute test set size.
    n_samples = x.shape[0]
    n_test = int(n_samples * prop_test)

    # Set up storage for true data and permuted data MSEs.
    d0_stats = np.zeros(num_perm)
    d1_stats = np.zeros(num_perm)
    
    # Create a regressor that scales logarithmically in xz_dim.
    clf = DecisionTreeRegressor(max_features='log2' if xz_dim > 10 else None)
    data_permutations = [np.random.permutation(n_samples) for i in range(num_perm)]

    datadict = {
            'x': x,
            'y': y,
            'z': z,
            'data_permutation': data_permutations,
            'n_test': n_test,
            'clf': clf,
            'reshuffle': True
            }
    
    # Compute mses for y = f(x, reshuffle(z)), varying train-test splits.
    d0_stats = np.array(joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
        joblib.delayed(obtain_error)((datadict, i)) for i in range(num_perm)))

    # Compute mses for y = f(x, z), varying train-test splits.
    datadict['reshuffle'] = False
    d1_stats = np.array(joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
        joblib.delayed(obtain_error)((datadict, i)) for i in range(num_perm)))

    if verbose:
        print('D0 statistics: {}'.format(d0_stats))
        print('D1 statistics: {}\n'.format(d1_stats))

    # Compute the p-value (one-tailed t-test
    # that mean of mse ratios equals 1).
    t, p_value = ttest_1samp(d0_stats / d1_stats, 1)
    if t < 0:
        p_value = 1 - p_value / 2
    else:
        p_value = p_value / 2

    if plot_return:
        return (p_value, d0_stats, d1_stats)
    else:
        return p_value
