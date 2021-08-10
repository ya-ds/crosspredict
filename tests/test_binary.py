import os
from collections import namedtuple

from hyperopt import fmin, tpe, Trials, space_eval
import numpy as np
import pandas as pd
import pytest
import requests
import yaml

from crosspredict.crossval import \
    CrossLightgbmModel, CrossXgboostModel, CrossCatboostModel
from crosspredict.iterator import Iterator
from crosspredict.report_binary import ReportBinary

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)

PARAMETERS_FPATH = 'tests/parameters.yml'


@pytest.fixture(scope='module')
def get_params():
    with open(PARAMETERS_FPATH) as f_in:
        test_param_dict = yaml.safe_load(f_in)

    return test_param_dict


@pytest.fixture(scope='module')
def onetwotrip_dataset():

    file_url = 'https://boosters.pro/api/ch/files/pub/onetwotrip_challenge_train.csv'
    file_path = 'tests/onetwotrip_challenge_train.csv'

    if not os.path.isfile(file_path):
        my_file = requests.get(file_url)
        with open(file_path, 'wb') as f_in:
            f_in.write(my_file.content)

    df = pd.read_csv(file_path)

    unique_clients = pd.Series(df['userid'].unique())
    test_users = unique_clients.sample(frac=0.2, random_state=0)
    val_idx = df['userid'].isin(test_users)
    test = df[val_idx].copy()
    train = df[~val_idx].copy()

    feature_name = df.columns.values

    col_to_delete_list = ['goal1', 'orderid', 'userid']

    for feature in col_to_delete_list:
        feature_name = np.delete(
            feature_name,
            np.argwhere(feature_name == feature)
        )

    Data = namedtuple(
        'Data',
        ['train', 'test', 'col_feature_list', 'col_client', 'col_target', 'col_cat_list']
    )

    result = Data(
        train=train,
        test=test,
        col_feature_list=feature_name,
        col_client='userid',
        col_target='goal1',
        col_cat_list=['field3', 'field2', 'field11', 'field23', 'field18', 'field20'],
    )

    return result


@pytest.fixture(scope='module')
def create_iterator(onetwotrip_dataset, get_params):
    params = get_params['iterator']

    iter_df = Iterator(col_client=onetwotrip_dataset.col_client, **params)

    return iter_df


@pytest.fixture(scope='module')
def lightgbm_fixture(onetwotrip_dataset, create_iterator, get_params):

    train, test = onetwotrip_dataset.train, onetwotrip_dataset.test

    params = get_params['lightgbm_params']
    model_params = get_params['model']
    iter_df = create_iterator

    model_class = CrossLightgbmModel(iterator=iter_df,
                                     feature_name=onetwotrip_dataset.col_feature_list,
                                     params=params,
                                     cols_cat=onetwotrip_dataset.col_cat_list,
                                     col_target=onetwotrip_dataset.col_target,
                                     **model_params)
    result = model_class.fit(train)

    return iter_df, model_class, result


@pytest.fixture(scope='module')
def xgboost_fixture(onetwotrip_dataset, create_iterator, get_params):

    train, test = onetwotrip_dataset.train, onetwotrip_dataset.test

    params = get_params['xgboost_params']
    model_params = get_params['model']
    iter_df = create_iterator

    model_class = CrossXgboostModel(iterator=iter_df,
                                    feature_name=onetwotrip_dataset.col_feature_list,
                                    params=params,
                                    cols_cat=onetwotrip_dataset.col_cat_list,
                                    col_target=onetwotrip_dataset.col_target,
                                    **model_params)
    result = model_class.fit(train)

    return iter_df, model_class, result


@pytest.fixture(scope='module')
def catboost_fixture(onetwotrip_dataset, create_iterator, get_params):

    train, test = onetwotrip_dataset.train, onetwotrip_dataset.test

    params = get_params['catboost_params']
    model_params = get_params['model']
    iter_df = create_iterator

    model_class = CrossCatboostModel(iterator=iter_df,
                                     feature_name=onetwotrip_dataset.col_feature_list,
                                     params=params,
                                     cols_cat=onetwotrip_dataset.col_cat_list,
                                     col_target=onetwotrip_dataset.col_target,
                                     **model_params)
    result = model_class.fit(train)

    return iter_df, model_class, result


def test_lightgbm_provide_appropriate_quality(lightgbm_fixture):
    iter_df, model_class, result = lightgbm_fixture
    assert result['loss'] < -0.69, 'test avg_loss'
    assert all(np.array(result['scores_all']) > 0.69), 'test each loss'


def test_xgboost_provide_appropriate_quality(xgboost_fixture):
    iter_df, model_class, result = xgboost_fixture
    assert result['loss'] < -0.69, 'test avg_loss'
    assert all(np.array(result['scores_all']) > 0.69), 'test each loss'


def test_catboost_provide_appropriate_quality(catboost_fixture):
    iter_df, model_class, result = catboost_fixture
    assert result['loss'] < -0.69, 'test avg_loss'
    assert all(np.array(result['scores_all']) > 0.69), 'test each loss'


def test_cross_catboost_model_can_get_hyperopt_space():
    hyperopt_space = CrossCatboostModel.get_hyperopt_space()
    assert type(hyperopt_space) == dict, (
        f'classmethod get_hyperopt_space returns object of type {type(hyperopt_space)},'
        f' but should return dict type'
    )


@pytest.mark.slow
def test_cross_catboost_work_with_hyperopt_correctly(onetwotrip_dataset, create_iterator, get_params, printer):
    printer(
        'WARNING! Slow test is running... (Use --skip-slow option to skip)'
    )

    train, test = onetwotrip_dataset.train, onetwotrip_dataset.test

    model_params = get_params['model']
    iter_df = create_iterator

    space = CrossCatboostModel.get_hyperopt_space()

    def score(params):
        cv_score = CrossCatboostModel(iterator=iter_df,
                                      feature_name=onetwotrip_dataset.col_feature_list,
                                      params=params,
                                      cols_cat=onetwotrip_dataset.col_cat_list,
                                      col_target=onetwotrip_dataset.col_target,
                                      **model_params)

        return cv_score.fit(train)

    trials = Trials()
    best = fmin(
        fn=score, space=space, algo=tpe.suggest, trials=trials, max_evals=1
    )
    result = space_eval(space, best)

    assert len(result.keys()) == len(space.keys())


def test_catboost_compatible_with_plot_report_functionality(onetwotrip_dataset, catboost_fixture):
    test = onetwotrip_dataset.test
    iter_df, model_class, result = catboost_fixture
    test['PREDICT'] = model_class.predict(test)

    report = ReportBinary()
    report.plot_report(
        df=test,
        cols_score=['PREDICT'],
        cols_target=[onetwotrip_dataset.col_target],
        report_shape=[1, 3],
        report={
            'Roc-Auc': {'loc': (0, 0)},
            'Precision-Recall': {'loc': (0, 1)},
            'Probability-Distribution': {'loc': (0, 2)}
        }
    )

    report.fig.savefig('test_fig.png')

    assert hasattr(report, 'stats')
    assert len(report.fig.axes) == 3
    assert os.path.isfile('test_fig.png')


def test_catboost_compatible_with_shap_functionality(onetwotrip_dataset, catboost_fixture):
    train = onetwotrip_dataset.train
    iter_df, model_class, result = catboost_fixture

    fig, shap_df = model_class.shap(train)
    fig.savefig('shap_test.png')

    assert os.path.isfile('shap_test.png')
