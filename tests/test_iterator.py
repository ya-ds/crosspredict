import requests
import os.path
import pandas as pd
import pytest
from crosspredict.iterator import Iterator
import numpy as np

@pytest.fixture
def iterator_fixture():
    file_url = 'https://boosters.pro/api/ch/files/pub/onetwotrip_challenge_train.csv'
    file_path = 'tests/onetwotrip_challenge_train.csv'
    if os.path.isfile(file_path)!=True:
        myfile = requests.get(file_url)
        open(file_path, 'wb').write(myfile.content)

    df = pd.read_csv('tests/onetwotrip_challenge_train.csv')

    n_repeats=3
    n_splits=10
    random_state = 0
    return df, dict(n_splits=n_splits,
                      n_repeats=n_repeats,
                      random_state=random_state)

def test_iterator_by_stratifiedall(iterator_fixture):
    df, dict_params = iterator_fixture

    iter_df = Iterator(**dict_params,
                       col_target = 'goal1',
                       cv_byclient=False)
    gen_df = iter_df.split(df)
    for i in range(dict_params['n_repeats']):
        repeat_ind = []
        for j in range(dict_params['n_splits']):
            X_train, X_val = next(gen_df)
            repeat_ind = repeat_ind + X_val.index.values.tolist()
            assert len(X_val)+len(X_train)==len(df)
        assert all(np.sort(np.unique(repeat_ind)) == np.sort(np.unique(df.index.values.tolist())))

def test_iterator_by_alldata(iterator_fixture):
    df, dict_params = iterator_fixture

    iter_df = Iterator(**dict_params,
                       cv_byclient=False)
    gen_df = iter_df.split(df)
    for i in range(dict_params['n_repeats']):
        repeat_ind = []
        for j in range(dict_params['n_splits']):
            X_train, X_val = next(gen_df)
            repeat_ind = repeat_ind + X_val.index.values.tolist()
            assert len(X_val)+len(X_train)==len(df)
        assert all(np.sort(np.unique(repeat_ind)) == np.sort(np.unique(df.index.values.tolist())))

def test_iterator_by_client(iterator_fixture):
    df, dict_params = iterator_fixture

    iter_df = Iterator(**dict_params,
                       col_client = 'userid',
                       cv_byclient=True)
    gen_df = iter_df.split(df)
    for i in range(dict_params['n_repeats']):
        repeat_ind = []
        for j in range(dict_params['n_splits']):
            X_train, X_val = next(gen_df)
            repeat_ind = repeat_ind + X_val.index.values.tolist()
            assert len(X_val)+len(X_train)==len(df)
        assert all(np.sort(np.unique(repeat_ind)) == np.sort(np.unique(df.index.values.tolist())))

