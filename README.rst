.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*A Decision Tree (Conditional) Independence Test (DTIT).*

Usage
-----
Let *x, y, z* be random variables. Then deciding whether *P(x | y, z) = P(x | z)* 
can be highly non-trivial, especially if the variables are continuous. This package 
implements a simple yet efficient and effective conditional independence test,
described in [link to arXiv when we write it up!]. Basic usage is simple:

.. code:: python 

    import dtit
    # Generate some data such that x is indpendent of y given z.
    n_samples = 300
    z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
    x = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    y = np.vstack([np.random.multinomial(20, p) for p in z])[:, :-1]
    
    # Run the conditional independence test.
    pval = dtit.test(x, y, z)

Here, we created discrete variables *x* and *y*, d-separated by a "common cause"
*z*. The null hypothesis is that *x* is independent of *y* given *z*. Since in this 
case the variables are independent given *z*, pval shouldn't be too small. Specifying which 
variables are discrete is optional.


Requirements
------------
To use the nn methods:
    * numpy >= 1.12
    * scikit-learn >= 0.18.1
    * scipy >= 0.16.1

.. _pip: http://www.pip-installer.org/en/latest/
