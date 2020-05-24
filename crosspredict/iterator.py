import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
import warnings


class Iterator:
    """
    K-Fold data with 3 different crossvalidation strategies:

    - crossvalidation by users (RepeatedKFold) should pass col_client and cv_byclient=True
    - stratified crossvalidation by target column (RepeatedStratifiedKFold) should pass col_target and cv_byclient=False
    - simple crossvalidation (RepeatedKFold) should pass col_target=None and cv_byclient=False

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.
    random_state : int or RandomState instance, default=0
        Pass an int for reproducible output across multiple
        function calls.
    col_target: str, default=None
        Column name for stratified crossvalidation
    col_client: str, default=None
        Column name for crossvalidation by users
    cv_byclient: bool, default=False
        flag if "crossvalidation by users" is needed
    """

    def __init__(self,
                 n_splits=5,
                 n_repeats=1,
                 random_state=0,
                 col_target=None,
                 col_client=None,
                 cv_byclient=False):
        """
        :param n_splits: int, default=5 Number of folds. Must be at least 2.
        :param n_repeats: int, default=1 Number of times cross-validator needs to be repeated.
        :param random_state: int or RandomState instance, default=0 Pass an int for reproducible output across multiple function calls.
        :param col_target: str, default=None Column name for stratified crossvalidation
        :param col_client: str, default=None Column name for crossvalidation by users
        :param cv_byclient: bool, default=False flag if "crossvalidation by users" is needed
        """

        if cv_byclient:
            assert col_client is not None, "You need provide col_client argument if cv_byclient=True (for RepeatedKFold by client)"
            warnings.warn(f"Using RepeatedKFold by column group \"{col_client}\"")
            self._repr = f"Using RepeatedKFold by column group \"{col_client}\""
        elif col_target is not None:
            warnings.warn(f"Using RepeatedStratifiedKFold by column group \"{col_target}\"")
            self._repr = f"Using RepeatedStratifiedKFold by column group \"{col_target}\""
        else:
            warnings.warn(f"Using RepeatedKFold by all data")
            self._repr = f"Using RepeatedKFold by all data"

        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.random_state = random_state

        self._df_len = None
        self.col_target = col_target
        self.col_client = col_client
        self.cv_byclient = cv_byclient

    def __repr__(self):
        return '<class Iterator> ' + self._repr

    def fit(self, df):

        self._df_len = len(df)

        if self.cv_byclient:
            self._unique_clients = df[self.col_client].unique()
            self._model_validation = RepeatedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state)
        elif self.col_target is not None:
            self._model_validation = RepeatedStratifiedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state)
        else:
            self._model_validation = RepeatedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state)

    def _get_split(self, df):
        if self.cv_byclient:
            return self._unique_clients.reshape(-1, 1),
        elif self.col_target is not None:
            return np.zeros((df.shape[0],1)), df[self.col_target]
        else:
            return np.zeros((df.shape[0],1)),

    def split(self, df):

        if self._df_len is None:
            self.fit(df)

        assert self._df_len == len(
            df), "Provided DataFrame doesn't match fitted DataFrame"

        for fold, (train_idx, val_idx) in enumerate(
                self._model_validation.split(*self._get_split(df))):
            if self.cv_byclient:
                val_idx = df[self.col_client].isin(
                    self._unique_clients[val_idx])
                train_idx = df[self.col_client].isin(
                    self._unique_clients[train_idx])
            elif self.col_target is not None:
                val_idx = df.index.isin(df.iloc[val_idx].index)
                train_idx = df.index.isin(df.iloc[train_idx].index)
            else:
                val_idx = df.index.isin(df.iloc[val_idx].index)
                train_idx = df.index.isin(df.iloc[train_idx].index)

            X_train, X_val = df.loc[train_idx], df.loc[val_idx]

            yield X_train, X_val
