.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*A Decision Tree (Conditional) Independence Test (DTIT).*

Introduction
-----------
Let *x, y, z* be random variables. Then deciding whether *P(y | x, z) = P(y | z)* 
can be difficult, especially if the variables are continuous. This package 
implements a simple yet efficient and effective conditional independence test,
described in [link to arXiv when we write it up!]. Important features that differentiate
this test from competition:

* It is fast. Worst-case speed scales as O(n_data * log(n_data) * dim), where dim is max(x_dim + z_dim, y_dim). However, amortized speed is O(n_data * log(n_data) * log(dim)).

* It applies to cases where some of x, y, z are continuous and some are discrete, or categorical (one-hot-encoded).

* It is very simple to understand and modify.

We have applied this test to tens of thousands of samples of thousand-dimensional datapoints in seconds. For smaller dimensionalities and sample sizes, it takes a fraction of a second. The algorithm is described in [arXiv link coming], where we also provide detailed experimental results and comparison with other methods. However for now, you should be able to just look through the code to understand what's going on -- it's only 90 lines of Python, including detailed comments!

Usage
-----
Basic usage is simple:
 
.. code:: python

  import numpy as np
  import dtit
  # Generate some data such that x is indpendent of y given z.
  n_samples = 300
  z = np.random.dirichlet(alpha=np.ones(2), size=n_samples)
  x = np.vstack([np.random.multinomial(20, p) for p in z])
  y = np.vstack([np.random.multinomial(20, p) for p in z])
  
  # Run the conditional independence test.
  pval = dtit.test(x, y, z)

Here, we created discrete variables *x* and *y*, d-separated by a "common cause"
*z*. The null hypothesis is that *x* is independent of *y* given *z*. Since in this 
case the variables are independent given *z*, pval should be distributed uniformly on [0, 1].

Requirements
------------
To use the nn methods:
    * numpy >= 1.12
    * scikit-learn >= 0.18.1
    * scipy >= 0.16.1

.. _pip: http://www.pip-installer.org/en/latest/