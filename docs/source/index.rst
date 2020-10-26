.. crosspredict documentation master file, created by
   sphinx-quickstart on Fri Feb 14 14:25:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CrossPredict's documentation!
========================================
.. image:: https://readthedocs.org/projects/crosspredict/badge/?version=latest 
   :target: https://crosspredict.readthedocs.io/en/latest/?badge=latest 
   :alt: Documentation Status

.. image:: https://badge.fury.io/py/crosspredict.png
   :target: https://badge.fury.io/py/crosspredict

* The library makes cross validation and reports generation easy
* Easy to extend to other models
* Supports Lightgbm, XGBoost
* Supports different crossvalidation strategies

    * Supports crossvalidation by users (RepeatedKFold)
    * Supports stratified crossvalidation by target column (RepeatedStratifiedKFold)
    * Supports simple crossvalidation (RepeatedKFold)
* Easy use of target encoding with double crossvalidation
* Supports target encoding library category_encoders


Table of contents
=================

- `Installation`_
- `Reports Preview`_
- `Target Encoding with DoubleCrossValidation`_
- `How to use`_
- `Authors`_

Installation
------------

::

    python -m pip install crosspredict

Reports Preview
---------------

::

    #create report object
    a = ReportBinary()

::

    a.plot_report(
       df,
       report_shape = (5,2),
       report={'Roc-Auc':  {'loc':(0, 0)},
             'Precision-Recall': [{'loc':(0, 1)}],
             'MeanTarget-by-Probability': [{'loc':(1, 0)},{'loc':(1, 1)}],
             'Gini-by-Generations': {'loc':(2,0), 'colspan':2},
             'MeanTarget-by-Generations': {'loc':(3,0), 'colspan':2},
             'Probability-Distribution': [{'loc':(4,0)},{'loc':(4,1)}]},
       cols_score = ['result_egr_to_one','probability'],
       cols_target = ['target','target2'],
       col_generation_deals='first_dt_no_comm_mon'
   )
    a.fig.savefig('report1.png')

.. image:: https://github.com/ya-ds/crosspredict/raw/master/notebooks/report1.png

::

    a.plot_report(report_shape = (4,2),
                  report={'Roc-Auc':  {'loc':(0, 0)},
                         'Precision-Recall': [{'loc':(0, 1)}],
                         'MeanTarget-by-Probability': [{'loc':(1, 0)}],
                         'Gini-by-Generations': {'loc':(2,0), 'colspan':2},
                         'MeanTarget-by-Generations': {'loc':(3,0), 'colspan':2},
                         'Probability-Distribution': [{'loc':(1,1)}]},
                  cols_score=['probability'])
    a.fig.savefig('report2.png')

.. image:: https://github.com/ya-ds/crosspredict/raw/master/notebooks/report2.png


Target Encoding with DoubleCrossValidation
------------------------------------------

::

    # creates iterator
    iter_df = Iterator(n_repeats=3,
                        n_splits=10,
                        random_state = 0,
                        col_client = 'userid',
                        cv_byclient=True)

    # fits target encoder (creates mappings for each fold)
    cross_encoder = CrossTargetEncoder(iterator = iter_df,
                                        encoder_class=WOEEncoder,
                                        n_splits= 5,
                                        n_repeats= 3,
                                        random_state= 0,
                                        col_client= 'userid',
                                        cv_byclient= True,
                                        col_encoded= 'goal1',
                                        cols= ['field3','field2','field11','field23','field18','field20']
                                      )
    cross_encoder.fit(train)

    # train cross validation models
    model_class = CrossLightgbmModel(iterator=iter_df,
                                     feature_name=feature_name,
                                     params=params,
                                     cols_cat = ['field3', 'field2', 'field11', 'field23', 'field18', 'field20'],
                                     num_boost_round = 9999,
                                     early_stopping_rounds = 50,
                                     valid = True,
                                     random_state = 0,
                                     col_target = 'goal1',
                                     cross_target_encoder = cross_encoder)
    result = model_class.fit(train)


How to use
----------

**Plot_Reports_for_Binary_Classification_problem_example** - `Plot_Reports_for_Binary_Classification_problem_example.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/Plot_Reports_for_Binary_Classification_problem_example.ipynb>`_

**Simple_example_in_one_Notebook** - `Simple_example_in_one_Notebook.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/Simple_example_in_one_Notebook.ipynb>`_

**Iterator_class** - `Iterator_class.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/Iterator_class.ipynb>`_

**CrossModelFabric_class** - `CrossModelFabric_class.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/CrossModelFabric_class.ipynb>`_

**CrossTargetEncoder_class** - `CrossTargetEncoder_class.ipynb <https://github.com//crosspredict/crosspredict/blob/master/notebooks/CrossTargetEncoder_class.ipynb>`_


Authors
-------

`Vladislav Boyadzhi <https://www.linkedin.com/in/vladislav-boyadzhi/>`_

Contents:
---------
.. toctree::
   :maxdepth: 2

   crossvalidation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
