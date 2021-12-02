from typing import Dict, Union
import pandas as pd
import logging
import json
from crosspredict.iterator import Iterator
from crosspredict.crossval import CrossLightgbmModel
import matplotlib.pyplot as plt
from jinja2 import Template
from hyperopt import fmin, tpe, Trials, space_eval
import lightgbm as lgb
import numpy as np
from crosspredict.report_binary._curves import CurveFabric
import os


def make_report(template, report):
    report = json.loads(report)
    t = Template(''.join(template))
    readme_rendered = t.render(**report)
    return readme_rendered

def _check_features(df,feature_name,cols_exclude,cols_cat,lgb_params):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    if feature_name:
        pass
    elif cols_exclude:
        feature_name = [col for col in df.columns if col not in cols_exclude]
    assert feature_name, 'You should pass either `feature_name` or `cols_exclude`'
    log.info('len(feature_name): ' + str(len(feature_name)))
    
    if cols_cat:
        cols_cat = [i for i in cols_cat if i in feature_name]
        log.info('len(cols_cat): '+ str(len(cols_cat)))
    else:
        log.info('len(cols_cat): 0')
        
        
    if lgb_params:
        if type(lgb_params) == str:
            lgb_params = json.loads(lgb_params)["params"]
    return feature_name,cols_exclude,cols_cat,lgb_params

def model_fit(df,
              lgb_params: Union[Dict, Union[str, bytes]],
              n_repeats,
              n_splits,
              num_boost,
              col_target,
              col_client,
              cv_byclient,
              cols_cat=None,
              cols_exclude=None,
              feature_name=None,
              early_stopping=50,
              **kwargs
              ):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    
    feature_name,cols_exclude,cols_cat,lgb_params = _check_features(df,feature_name,cols_exclude,cols_cat,lgb_params)
    
    params = lgb_params

    iter_df = Iterator(n_repeats=n_repeats,
                       n_splits=n_splits,
                       random_state=0,
                       col_target=col_target,
                       cv_byclient=cv_byclient,
                       col_client=col_client)

    model_class = CrossLightgbmModel(iterator=iter_df,
                                     feature_name=feature_name,
                                     params=params,
                                     num_boost_round=num_boost,
                                     early_stopping_rounds=early_stopping,
                                     valid=True,
                                     random_state=0,
                                     col_target=col_target,
                                     cols_cat=cols_cat
                                     )

    result = model_class.fit(df)
    log.info(result)
    fig, shap_df = model_class.shap(df)

    report = {'col_target': col_target,
              'score_max': format(float(result['score_max']), '.5f'),
              'scores_all': [format(float(i), '.5f') for i in result['scores_all']],
              'frac': [format(float(i), '.5f') for i in result['scores_all']],
              'loss': format(float(result['loss']), '.5f'),
              'std': format(float(result['std']), '.5f'),
              'shap_df': shap_df[:15].to_markdown(),
              'feature_len': str(len(feature_name)),
              'num_boost': str(result['num_boost'])}
    report = json.dumps(report, indent=4, sort_keys=True)
    plt.tight_layout(True)
    return [fig, shap_df, report, model_class]


def forward_selection(df,
                      n_repeats,
                      n_splits,
                      col_target,
                      col_client,
                      cv_byclient,
                      cols_exclude,
                      cols_cat,
                      feature_name,
                      **kwargs):

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    feature_name,cols_exclude,cols_cat,_ = _check_features(df,feature_name,cols_exclude,cols_cat,None)

    iter_df = Iterator(n_repeats=n_repeats,
                       n_splits=n_splits,
                       random_state=0,
                       col_target=col_target,
                       cv_byclient=cv_byclient,
                       col_client=col_client)

    scores = []
    selected_features = []

    log.info("Starting force feature selection ...")
    for _ in range(len(feature_name)):

        top_feature = None
        top_score = 0
        
        for i, f in enumerate(set(feature_name) - set(selected_features)):
            current_features = selected_features + [f]
            log.info(current_features)
            _,_,cols_cat,_ = _check_features(df,current_features,None,cols_cat,None)
            model_class = CrossLightgbmModel(iterator=iter_df,
                                             feature_name=current_features,
                                             params={
                                                 'objective': 'binary',
                                                 'metric': 'auc',
                                                 'bagging_seed': 0,
                                                 'data_random_seed': 0,
                                                 'drop_seed': 0,
                                                 'feature_fraction_seed': 0,
                                                 'verbose': -1
                                             },
                                             num_boost_round=100,
                                             early_stopping_rounds=100,
                                             valid=True,
                                             random_state=0,
                                             col_target=col_target,
                                             cols_cat=cols_cat
                                             )


            result = model_class.fit(df)
            score = result['score_max']

            if score > top_score:
                top_score = score
                top_feature = f

        log.info("{} features left to select ...".format(len(set(feature_name) - set(selected_features)) - 1))
        scores.append(top_score)
        selected_features.append(top_feature)
            
        log.info(top_score)
        log.info(top_feature)
        plt.plot(scores)
        plt.show()

    log.info(scores)
    log.info(selected_features)

    plt.plot(scores)
    plt.show()

    scores_df = pd.DataFrame(scores, columns=["score"])
    scores_df = pd.DataFrame(zip(selected_features, scores), columns=["index","score"])
    
    scores_df['score2'] = scores_df['score'].shift(1)
    scores_df['score_diff'] = scores_df['score'] - scores_df['score2']
    scores_df['score_diff_flag'] = scores_df['score_diff'] < 0.00005
    
    top_features = scores_df[scores_df['score_diff_flag']].index[0]

    top_features = json.dumps({"feature_selection": str(top_features)}, indent=4, sort_keys=True)
    return [scores_df, top_features]


def onefactor(df_adv, shap_df, col_date, col_target, report_directory, output_file='README_onefactor.md', top_features=20):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    report_text=''
    for row in shap_df[:top_features].iterrows():
        text='\n'
        text += f'Feature name = {row[1]["feature"]}  \n'
        text += f'Shap Value = {row[1]["mean"]}  \n'
        text += f'![onefactor]({row[1]["feature"]}.png)  \n'
        report_text = report_text + text
        log.info(text)
        col_name = row[1]['feature']

        n_groups = 4


        splits = np.arange(0, n_groups + 1) / n_groups
        if df_adv[col_name].nunique() > n_groups:
            a = pd.qcut(df_adv[col_name], splits, duplicates='drop')
        else:
            a = df_adv[col_name].astype('category')
        a.cat.add_categories(['missing'], inplace=True)
        a = a.fillna('missing')

        fig, _ = plt.subplots()
        fig.set_size_inches(1 * 8, 2 * 5)
        plt.subplots_adjust(left=0.125,
                            right=0.9,
                            bottom=0.1,
                            top=0.95,
                            wspace=0.1,
                            hspace=0.3
                            )

        b = df_adv.groupby([col_date, a]).size().unstack(level=1)
        b = b.div(b.sum(axis=1), axis=0)
        c = df_adv.groupby([df_adv[col_date]]).size()

        ax1 = plt.subplot2grid((2, 1), loc=(0, 0))
        ax2 = ax1.twinx()
        for col in b.columns:
            CurveFabric._draw_series_to_line_plot(_, b[col], ax=ax2, label=col, name=col, legend=True)
        CurveFabric._draw_series_to_bar_plot(_, c, ax=ax1)
        ax2.set_ylabel('Count, %')

        b = df_adv.groupby([col_date, a])[col_target].mean().unstack(level=1)
        c = df_adv.groupby([col_date]).size()

        ax3 = plt.subplot2grid((2, 1), loc=(1, 0))
        ax4 = ax3.twinx()
        for col in b.columns:
            CurveFabric._draw_series_to_line_plot(_, b[col], ax=ax4, label=col, name=col, legend=True)
        CurveFabric._draw_series_to_bar_plot(_, c, ax=ax3)
        ax4.set_ylabel('Mean Target')

        ax2.set_title(col_name, fontsize=14, fontweight='bold')
        fig.savefig(os.path.join(report_directory, col_name + '.png'))

    with open(os.path.join(report_directory, output_file), 'w') as f:
        f.write(report_text)
        
    return report_text
    


def hyperopt_fit(df,n_repeats,n_splits,hyperopt_trials,num_boost,col_target, col_client,cv_byclient, cols_cat=None, feature_name=None,cols_exclude=None,**kwargs):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    feature_name,cols_exclude,cols_cat,_ = _check_features(df,feature_name,cols_exclude,cols_cat,None)

    space = CrossLightgbmModel(iterator=None,
                               feature_name=feature_name,
                               params=None,
                               num_boost_round=num_boost,
                               early_stopping_rounds=50,
                               valid=True,
                               random_state=0,
                               col_target=col_target).get_hyperopt_space()

    iter_df = Iterator(n_repeats=n_repeats,
                       n_splits=n_splits,
                       random_state=0,
                       col_target=col_target,
                       cv_byclient=cv_byclient,
                       col_client=col_client)

    def score(params):
        params['njobs'] = 8
        cv_score = CrossLightgbmModel(iterator=iter_df,
                                      feature_name=feature_name,
                                      params=params,
                                      num_boost_round=9999,
                                      early_stopping_rounds=50,
                                      valid=True,
                                      random_state=0,
                                      col_target=col_target)
        return cv_score.fit(df=df)

    trials = Trials()
    best = fmin(fn=score,
                space=space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=hyperopt_trials
                )

    results = {"params": space_eval(space, best)}
    results["hyperopt_trials"] = hyperopt_trials

    results = json.dumps(results, indent=4, sort_keys=True)
    return results


def model_single_fit(df, lgb_params, col_target, num_boost, cols_cat=None, cols_exclude=None, feature_name=None, **kwargs):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    feature_name,cols_exclude,cols_cat,_ = _check_features(df,feature_name,cols_exclude,cols_cat,None)

    dataset = lgb.Dataset(
        data=df[feature_name],
        label=df[col_target],
        categorical_feature=cols_cat)
    log.info('feature_num: ' + str(len(feature_name)))
    log.info('num_boost: ' + str(num_boost))
    log.info(str(lgb_params))
    model = lgb.train(lgb_params,
                      dataset,
                      num_boost_round=num_boost)

    model_str = model.model_to_string()

    return model_str, model
