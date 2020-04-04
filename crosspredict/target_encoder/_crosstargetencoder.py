import pandas as pd
from crosspredict.target_encoder import TargetEncoder


class CrossTargetEncoder:
    def __init__(self,
                 iterator,
                 encoder_class,
                 cols,
                 col_encoded,
                 n_splits=10,
                 n_repeats=1,
                 random_state=0,
                 col_target=None,
                 col_client=None,
                 cv_byclient=False,
                 **kwargs):
        self.iterator = iterator

        self._encoder_class = encoder_class
        self._encoder_kwargs = kwargs
        self.cols = cols
        self.col_encoded = col_encoded

        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.random_state = random_state

        self.col_target = col_target
        self.col_client = col_client
        self.cv_byclient = cv_byclient

        self._targetencoder_list = []
        self._targetencoded_cols = ['encoded_' + col for col in self.cols]

    def fit(self, df):
        self._targetencoder_list = []
        for X_train, X_val in self.iterator.split(df):
            encoder_ = TargetEncoder(encoder_class=self._encoder_class,
                                     cols=self.cols,
                                     col_encoded=self.col_encoded,
                                     n_splits=self.n_splits,
                                     n_repeats=self.n_repeats,
                                     random_state=self.random_state,
                                     col_target=self.col_target,
                                     col_client=self.col_client,
                                     cv_byclient=self.cv_byclient,
                                     **self._encoder_kwargs)
            encoder_.fit(X_train)
            self._targetencoder_list.append(encoder_)

    def transform(self, fold, train=None, test=None):
        encoder_ = self._targetencoder_list[fold]
        assert (train is not None) | (test is not None)

        transformed_test = None
        transformed_train = None

        if train is None:
            transformed_test = encoder_.predict(test)
        elif test is None:
            transformed_train = encoder_.transform(train)
        else:
            transformed_test = encoder_.predict(test)
            transformed_train = encoder_.transform(train)

        return transformed_train, transformed_test

    def predict(self, df):
        encoder_count = 0
        result = pd.DataFrame(
            index=df.index, columns=self._targetencoded_cols, data=0)
        for encoder_ in self._targetencoder_list:
            result += encoder_.predict(df) * \
                encoder_.n_splits * encoder_.n_repeats
            encoder_count += encoder_.n_splits * encoder_.n_repeats
        result = result / encoder_count
        return result
