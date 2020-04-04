.. crosspredict documentation master file, created by
   sphinx-quickstart on Fri Feb 14 14:25:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CrossPredict's documentation!
========================================

* The library makes cross validation easy
* Easy use of target encoding with double crossvalidation
* Easy to extend to other models
* Supports Lightgbm, XGBoost
* Supports crossvalidation by users (RepeatedKFold)
* Supports stratified crossvalidation by target column (RepeatedStratifiedKFold)
* Supports simple crossvalidation (RepeatedKFold)
* Supports target encoding library category_encoders


Table of contents
-----------------

- [Installation](#installation) :ref:`installation`
- [How to use](#how-to-use)

.. _installation:
Installation
------------

::

   git clone https://github.com/crosspredict/crosspredict
   python -m pip install .


How to use
----------

**Simple_example_in_one_Notebook** - `Simple_example_in_one_Notebook.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/Simple_example_in_one_Notebook.ipynb>`_

**Iterator_class** - `Iterator_class.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/Iterator_class.ipynb>`_

**CrossModelFabric_class** - `CrossModelFabric_class.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/CrossModelFabric_class.ipynb>`_

**CrossTargetEncoder_class** - `CrossTargetEncoder_class.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/CrossTargetEncoder_class.ipynb>`_

Contents:
--------
.. toctree::
   :maxdepth: 2

   crossvalidation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
