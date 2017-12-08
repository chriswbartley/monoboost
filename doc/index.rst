.. monoboost documentation master file, created by sphinx-quickstart on Tue Apr 14 10:29:06 2015. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

.. role:: bash(code)
   :language: bash

Welcome to monoboost's documentation!
====================================

`Monoboost` is the first instance based classification algorithm that allows for *partial* monotonicity (i.e. some *non*-monotone features). It uses standard inequality constraints on the monotone features, and novel L1 cones to place sensible constraints on non-monotone features.

To install, simply use :bash:`pip install monoboost`. For full documentation you've come to the right place. For a brief overview, refer to the `README file 
<https://github.com/chriswbartley/monoboost/blob/master/README.md>`_ in the Github repository.

Contents:

.. toctree::
   :maxdepth: 2

   theory
   auto_examples/index
   api
