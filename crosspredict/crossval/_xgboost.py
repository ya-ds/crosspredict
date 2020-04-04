from __future__ import annotations
from hyperopt import hp
from hyperopt.pyll import scope
import xgboost as xgb
from ._crossval import CrossModelFabric

class CrossXgboostModel(CrossModelFabric):
    def get_hyperopt_space(self, params={}, random_state=None):
        if random_state is None:
            random_state = self.random_state
        result = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
            'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth': scope.int(hp.quniform('max_depth', 1, 14, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
            # 'nthread': 4,
            'booster': 'gbtree',
            'tree_method': 'exact',
            'silent': 1,
            'seed': random_state
        }
        if params != {}:
            result.update(params)
        return result

    def get_dataset(self, data, label, categorical_feature, **kwargs):
        return xgb.DMatrix(data=data, label=label, **kwargs)

    def train(
            self,
            params,
            train_set,
            train_name,
            valid_set,
            valid_name,
            num_boost_round,
            evals_result,
            categorical_feature,
            early_stopping_rounds,
            verbose_eval,
            **kwargs):
        model = xgb.train(params=params,
                          dtrain=train_set,
                          evals=[(train_set, train_name),
                                 (valid_set, valid_name)],
                          num_boost_round=num_boost_round,
                          evals_result=evals_result,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval,
                          **kwargs)
        return model
