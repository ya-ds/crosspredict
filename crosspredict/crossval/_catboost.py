from typing import Optional, Union, List
import sys

from hyperopt import hp
from hyperopt.pyll import scope
import catboost as cb
from catboost import Pool
from torch import cuda

from ._crossval import CrossModelFabric


class CrossCatboostModel(CrossModelFabric):
    @classmethod
    def get_hyperopt_space(
            cls,
            params: Optional[dict] = None,
            random_state: int = 0
    ) -> dict:

        result = {
            'max_depth': scope.int(hp.quniform('max_depth', 6, 16, 1)),
            'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 10, 150, 1)),
            'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
            'random_strength': hp.uniform('random_strength', 0.0, 100),
            'rsm': hp.uniform('feature_fraction', 0.75, 1.0),
            'l2_leaf_reg': hp.uniform('lambda_l2', 1e-4, 10),
            'leaf_estimation_iterations': scope.int(hp.quniform('leaf_estimation_iterations', 1, 20, 1)),
            'leaf_estimation_backtracking': hp.choice('leaf_estimation_backtracking', ['No', 'AnyImprovement']),
            'random_state': random_state,
            'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
            'eval_metric': 'AUC',
            'verbose': 200,
            'loss_function': 'Logloss',
        }

        if params:
            result.update(params)

        return result

    @staticmethod
    def get_dataset(
            data,
            label,
            categorical_feature: List[str],
            **kwargs
    ) -> Pool:

        if categorical_feature == 'auto':

            print('WARNING: auto categorical features detection is not supported.'
                  'All features is considered as numerical', file=sys.stderr)

            categorical_feature = []

        data[categorical_feature] = data[categorical_feature].astype('int64')

        dataset = Pool(
            data=data,
            label=label,
            cat_features=categorical_feature,
            **kwargs
        )
        return dataset

    def train(
            self,
            params: dict,
            train_set: Pool,
            train_name: Optional[str],
            valid_set: Union[Pool, List[Pool]],
            valid_name: Optional[str],
            num_boost_round: int,
            evals_result: dict,
            categorical_feature: List[str],
            early_stopping_rounds: int,
            verbose_eval: bool,
            **kwargs
    ) -> cb.core.CatBoost:

        # for compatibility with CrossModelFabric API
        if 'metric' in params.keys():
            params.pop('metric')    # Catboost haven't param 'metric', 'eval_metric' instead

        # auto switch to GPU task type if cuda is available
        if 'task_type' not in params.keys() and cuda.is_available():
            params['task_type'] = 'GPU'

        # train Catboost model
        _model = cb.train(params=params,
                          dtrain=train_set,
                          eval_set=valid_set,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval,
                          **kwargs)

        # for compatibility with CrossModelFabric API
        model = _model.copy()
        params['metric'] = params['eval_metric']
        model.feature_name = lambda: _model.feature_names_
        type_dict = {x: int for x in categorical_feature}

        if len(model.classes_) == 2:
            model.predict = lambda df, num_iteration: \
                _model.predict(
                    df.astype(dtype=type_dict),
                    ntree_end=num_iteration,
                    prediction_type='Probability'
                )[:, 1]
        else:
            model.predict = lambda df, num_iteration: \
                _model.predict(
                    df.astype(dtype=type_dict),
                    ntree_end=num_iteration,
                    prediction_type='Probability'
                )

        evals_result['eval'] = {
            params['eval_metric']:
                _model.evals_result_['validation'][params['eval_metric']][:-early_stopping_rounds]
        }

        return model
