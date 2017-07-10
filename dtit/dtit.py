""" A conditional independence test based on decision tree regression.

Reference:
Chalupka, Krzysztof and Perona, Pietro and Eberhardt, Frederick, 2017.
"""
import os
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse


def test(x, y, z=None, num_perm=10, prop_test=.1,
    discrete=(False, False), plot_return=False, verbose=False, **kwargs):
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
        kwargs: Arguments to pass to the neural net constructor.

    Returns:
        p (float): The p-value for the null hypothesis
            that x is independent of y.
    """
    # If x xor y is discrete, use the continuous variable as input.
    if discrete[0] and not discrete[1]:
        x, y = y, x

    # Otherwise, predict the variable with fewer dimensions.
    elif x.shape[1] < y.shape[1]:
        x, y = y, x

    y /= y.flatten().std()

    # Use this many datapoints as a test set.
    n_samples = x.shape[0]
    n_test = int(n_samples * prop_test)

    # Attach the conditioning variable to the input.
    if z is not None:
        x_z = np.hstack([x, z])
    else:
        x_z = x
    xz_dim = x_z.shape[1]

    # Set up storage for true data and permuted data MSEs.
    d0_stats = np.zeros(num_perm)
    d1_stats = np.zeros(num_perm)
    
    # Create a regressor that scales logarithmically in xz_dim.
    clf = DecisionTreeRegressor(max_features='log2' if xz_dim > 10 else None)
    
    for perm_id in range(num_perm):
        # Estimate MSE with permuted X.
        data_permutation = np.random.permutation(n_samples)
        perm_ids = np.random.permutation(n_samples)
        if z is not None:
            x_z_bootstrap = np.hstack([x[perm_ids], z])
        else:
            x_z_bootstrap = x[perm_ids]
        clf.fit(x_z_bootstrap[data_permutation][n_test:],
                y[data_permutation][n_test:])
        d0_stats[perm_id] = mse(y[data_permutation][:n_test],
            clf.predict(x_z_bootstrap[data_permutation][:n_test]))

        # Estimate the MSE with original X.
        clf.fit(x_z[data_permutation][n_test:], y[data_permutation][n_test:])
        d1_stats[perm_id] = mse(y[data_permutation][:n_test],
            clf.predict(x_z[data_permutation][:n_test]))

        if verbose:
            print('D0 statistic, iter {}: {}'.format(
                perm_id, d0_stats[perm_id]))
            print('D1 statistic, iter {}: {}\n'.format(
                perm_id, d1_stats[perm_id]))

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
