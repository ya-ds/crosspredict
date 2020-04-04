import pytest
import yaml

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold
import pandas as pd
pd.set_option('display.max_columns',999)
pd.set_option('display.max_rows',999)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from crosspredict.crossval import CrossLightgbmModel, CrossXgboostModel
from crosspredict.iterator import Iterator
import os
import requests

@pytest.fixture
def lightgbm_fixture():

    file_url = 'https://boosters.pro/api/ch/files/pub/onetwotrip_challenge_train.csv'
    file_path = 'tests/onetwotrip_challenge_train.csv'
    if os.path.isfile(file_path) != True:
        myfile = requests.get(file_url)
        open(file_path, 'wb').write(myfile.content)

    df = pd.read_csv(file_path)

    unique_clients = pd.Series(df['userid'].unique())
    test_users = unique_clients.sample(frac=0.2, random_state=0)
    val_idx = df['userid'].isin(test_users)
    test = df[val_idx].copy()
    train = df[~val_idx].copy()

    feature_name = df.columns.values
    feature_name = np.delete(feature_name, np.argwhere(feature_name == 'goal1'))
    feature_name = np.delete(feature_name, np.argwhere(feature_name == 'orderid'))
    feature_name = np.delete(feature_name, np.argwhere(feature_name == 'userid'))

    params = {'bagging_fraction': 0.849285747554019,
              'bagging_freq': 5,
              'bagging_seed': 0,
              'boosting_type': 'gbdt',
              'data_random_seed': 0,
              'drop_seed': 0,
              'feature_fraction': 0.8212766928844304,
              'feature_fraction_seed': 0,
              'lambda_l1': 0.8955546599539566,
              'lambda_l2': 1.4423261095989717,
              'learning_rate': 0.03,
              'max_bin': 255,
              'max_depth': 43,
              'metric': 'auc',
              'min_data_in_leaf': 149,
              'min_sum_hessian_in_leaf': 1.804477623298885,
              'num_leaves': 363,
              'objective': 'binary',
              'seed': 0,
              'verbose': -1}

    iter_df = Iterator(n_repeats=2,
                       n_splits=3,
                       random_state=0,
                       col_client='userid',
                       cv_byclient=True)

    model_class = CrossLightgbmModel(iterator=iter_df,
                                     feature_name=feature_name,
                                     params=params,
                                     cols_cat=['field3', 'field2', 'field11', 'field23', 'field18', 'field20'],
                                     num_boost_round=9999,
                                     early_stopping_rounds=50,
                                     valid=True,
                                     random_state=0,
                                     col_target='goal1', )
    result = model_class.fit(train)

    return iter_df, model_class, result

@pytest.fixture
def xgboost_fixture():

    file_url = 'https://boosters.pro/api/ch/files/pub/onetwotrip_challenge_train.csv'
    file_path = 'tests/onetwotrip_challenge_train.csv'
    if os.path.isfile(file_path) != True:
        myfile = requests.get(file_url)
        open(file_path, 'wb').write(myfile.content)

    df = pd.read_csv(file_path)

    unique_clients = pd.Series(df['userid'].unique())
    test_users = unique_clients.sample(frac=0.2, random_state=0)
    val_idx = df['userid'].isin(test_users)
    test = df[val_idx].copy()
    train = df[~val_idx].copy()

    feature_name = df.columns.values
    feature_name = np.delete(feature_name, np.argwhere(feature_name == 'goal1'))
    feature_name = np.delete(feature_name, np.argwhere(feature_name == 'orderid'))
    feature_name = np.delete(feature_name, np.argwhere(feature_name == 'userid'))

    params = {
        'max_depth': 4,
        'min_child_weight': 6,
        'gamma': 0.05,
        'colsample_bytree': 1,
        'subsample': 0.6,
        'scale_pos_weight': 1,
        'objective': 'binary:logistic',
        'eta': 0.1,
        'alpha': 0.9,
        'lambda': 0.6,
        'eval_metric': 'auc',
        'silent': 1,
        'verbose_eval': False,
        'seed': 0}

    iter_df = Iterator(n_repeats=2,
                       n_splits=3,
                       random_state=0,
                       col_client='userid',
                       cv_byclient=True)

    model_class = CrossXgboostModel(iterator=iter_df,
                                     feature_name=feature_name,
                                     params=params,
                                     cols_cat=['field3', 'field2', 'field11', 'field23', 'field18', 'field20'],
                                     num_boost_round=9999,
                                     early_stopping_rounds=50,
                                     valid=True,
                                     random_state=0,
                                     col_target='goal1', )
    result = model_class.fit(train)

    return iter_df, model_class, result

def test_lightgbm_avg_loss(lightgbm_fixture):
    iter_df, model_class, result = lightgbm_fixture
    assert result['loss']<-0.69

def test_lightgbm_each_loss(lightgbm_fixture):
    iter_df, model_class, result = lightgbm_fixture
    assert all(np.array(result['scores_all'])>0.69)==True


def test_xgboost_avg_loss(xgboost_fixture):
    iter_df, model_class, result = xgboost_fixture
    assert result['loss']<-0.69

def test_xgboost_each_loss(xgboost_fixture):
    iter_df, model_class, result = xgboost_fixture
    assert all(np.array(result['scores_all'])>0.69)==True