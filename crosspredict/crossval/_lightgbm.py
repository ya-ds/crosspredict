
from __future__ import annotations
from hyperopt import hp
from hyperopt.pyll import scope
import lightgbm as lgb
from ._crossval import CrossModelFabric

class CrossLightgbmModel(CrossModelFabric):
    def get_hyperopt_space(self, params={}, random_state=None):
        if random_state is None:
            random_state = self.random_state
        result = {
            'num_leaves': scope.int(hp.quniform('num_leaves', 100, 500, 1)),
            'max_depth': scope.int(hp.quniform('max_depth', 10, 70, 1)),
            'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 10, 150, 1)),
            'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
            'lambda_l1': hp.uniform('lambda_l1', 1e-4, 2),
            'lambda_l2': hp.uniform('lambda_l2', 1e-4, 2),
            'seed': random_state,
            'feature_fraction_seed': random_state,
            'bagging_seed': random_state,
            'drop_seed': random_state,
            'data_random_seed': random_state,
            'verbose': -1,
            'bagging_freq': 5,
            'max_bin': 255,
            'learning_rate': 0.03,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
        }
        if params != {}:
            result.update(params)
        return result

    def get_dataset(self, data, label, categorical_feature, **kwargs):
        return lgb.Dataset(
            data=data,
            label=label,
            categorical_feature=categorical_feature,
            **kwargs)

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
        model = lgb.train(params=params,
                          train_set=train_set,
                          valid_sets=[train_set, valid_set],
                          valid_names=[train_name, valid_name],
                          num_boost_round=num_boost_round,
                          evals_result=evals_result,
                          categorical_feature=categorical_feature,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval,
                          **kwargs)
        return model
