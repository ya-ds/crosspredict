from typing import List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from ._curves import PrecisionRecallCurve, RocAucCurve, MeanProbCurve, GenGINICurve, DistributionCurve, GenRiskCurve
import logging



class ReportBinary:
    _name_map = {
        'mean-prob': 'MeanTarget-by-Probability',
        'roc-auc': 'Roc-Auc',
        'precision-recall': 'Precision-Recall',
        'gen-gini': 'Gini-by-Generations',
        'gen-risk': 'MeanTarget-by-Generations',
        'distribution': 'Probability-Distribution'
    }

    _class_map = {
        'Roc-Auc': RocAucCurve,
        'Precision-Recall': PrecisionRecallCurve,
        'MeanTarget-by-Probability': MeanProbCurve,
        'Gini-by-Generations': GenGINICurve,
        'MeanTarget-by-Generations': GenRiskCurve,
        'Probability-Distribution': DistributionCurve
    }

    def __init__(self):
        self._stats = {}
    @property
    def stats(self):
        return pd.DataFrame(self._stats, index=['class']).T

    @property
    def stats_roc_auc(self):
        return self.stats.loc['Roc-Auc']['class'].map(lambda x: x.roc_auc)

    @classmethod
    def plot_roc_auc(cls,df, col_score, col_target, ax=None, title=None, **kwargs):
        return cls._plot_graph_class(df, col_score, col_target, report="Roc-Auc", ax=ax, title=title, **kwargs)

    @property
    def stats_precision_recall(self):
        return self.stats.loc['Precision-Recall']['class'].map(lambda x: x.average_precision)

    @classmethod
    def plot_precision_recall(cls,df, col_score, col_target, ax=None, title=None, **kwargs):
        return cls._plot_graph_class(df, col_score, col_target, report="Precision-Recall", ax=ax, title=title, **kwargs)

    @property
    def stats_gini_by_generations(self):
        return self.stats.loc['Gini-by-Generations']['class'].apply(lambda x: x.series_auc).T

    @classmethod
    def plot_gini_by_generations(cls,df, col_score, col_target, col_generation_deals, ax=None, title=None, **kwargs):
        return cls._plot_graph_class(df, col_score, col_target, report="Gini-by-Generations", ax=ax, title=title,
                                     col_generation_deals=col_generation_deals, **kwargs)

    @property
    def stats_meantarget_by_generations(self):
        return self.stats.loc['MeanTarget-by-Generations']['class'].apply(lambda x: x.risk['mean']).T

    @classmethod
    def plot_meantarget_by_generations(cls,df, col_score, col_target, col_generation_deals, ax=None, title=None, **kwargs):
        return cls._plot_graph_class(df, col_score, col_target, report="MeanTarget-by-Generations", ax=ax, title=title,
                                     col_generation_deals=col_generation_deals, **kwargs)

    def stats_meantarget_by_probability(self, col_score):
        return self.stats.loc['MeanTarget-by-Probability'].loc[col_score]['class'].apply(lambda x: x.mean_prob['mean']).T

    @classmethod
    def plot_meantarget_by_probability(cls,df, col_score, col_target, ax=None, title=None, **kwargs):
        return cls._plot_graph_class(df, col_score, col_target, report="MeanTarget-by-Probability", ax=ax, title=title, **kwargs)

    def _get_loc(self, locations, title):
        if (isinstance(locations, tuple)) | (isinstance(locations, list)):
            return locations
        else:
            return [locations]

    def plot_report(self,
                    df: pd.DataFrame,
                    report_shape: List[int],
                    report: Dict,
                    cols_score: List[str],
                    cols_target: List[str],
                    col_generation_deals: str=None,
                    col_generation_apps: str=None):
        """
        Plots report of given configuration.

        :param report_shape: List[int] Shape of subplot axes. Read more https://matplotlib.org/3.1.1/gallery/userdemo/demo_gridspec01.html#sphx-glr-gallery-userdemo-demo-gridspec01-py
        :param report: Dict Dict with reports and their location. Read more https://matplotlib.org/3.1.1/gallery/userdemo/demo_gridspec01.html#sphx-glr-gallery-userdemo-demo-gridspec01-py
        :param cols_score: List[str] SubList of column names with model probabilities
        :param cols_target: List[str] SubList of column names with true binary labels
        :return:
        """
        log = logging.getLogger(__name__)
        log.info(f"DataFrame's shape = {df.shape}")
        height = report_shape[0]
        width = report_shape[1]

        self.fig = plt.gcf()
        self.fig.set_size_inches(width * 6, height * 5)
        plt.subplots_adjust(left=0.125,
                            right=0.9,
                            bottom=0.1,
                            top=0.95,
                            wspace=0.35,
                            hspace=0.6
                            )

        mask_score = ~(df[cols_score].isnull().any(axis=1))
        mask_target = ~(df[cols_target].isnull().any(axis=1))

        for report, locations in report.items():

            if report in self._name_map.keys():
                report = self._name_map[report]

            if report == 'Roc-Auc':
                log.info(f"{report} plot uses DataFrame with score and target. DataFrame's shape = {df[mask_score & mask_target].shape}")
                loc = self._get_loc(locations, report)
                assert len(loc) == 1, f'Location of {report} plot should have `len`==1, passed `len`={len(loc)}'
                ax = plt.subplot2grid((height, width), **loc[0])
                for col_target, col_score in product(cols_target, cols_score):
                    self._plot_graph(df[mask_score & mask_target],
                                     col_score,
                                     col_target,
                                     report=report,
                                     ax=ax,
                                     title=report)

            elif report == 'Precision-Recall':
                log.info(f"{report} plot uses DataFrame with score and target. DataFrame's shape = {df[mask_score & mask_target].shape}")
                loc = self._get_loc(locations, report)
                assert len(loc) == 1, f'Location of {report} plot should have `len`==1, passed `len`={len(loc)}'
                ax = plt.subplot2grid((height, width), **loc[0])
                for col_target, col_score in product(cols_target, cols_score):
                    self._plot_graph(df[mask_score & mask_target],
                                     col_score,
                                     col_target,
                                     report=report,
                                     ax=ax,
                                     title=report)

            elif report == 'Gini-by-Generations':
                log.info(f"{report} plot uses all DataFrame. DataFrame's shape = {df.shape}")
                loc = self._get_loc(locations, report)
                assert len(loc) == 1, f'Location of {report} plot should have `len`==1, passed `len`={len(loc)}'
                assert col_generation_deals is not None, f'To plot GINI by generations you need to pass `col_generation_deals`'
                ax = plt.subplot2grid((height, width), **loc[0])
                ax_twinx = ax.twinx()
                for col_target, col_score in product(cols_target, cols_score):
                    self._plot_graph(df,
                                     col_score,
                                     col_target,
                                     ax=ax,
                                     report=report,
                                     title=report,
                                     ax_twinx=ax_twinx,
                                     col_generation_deals=col_generation_deals)

            elif report == 'MeanTarget-by-Generations':
                log.info(f"{report} plot uses all DataFrame. DataFrame's shape = {df.shape}")
                loc = self._get_loc(locations, report)
                assert len(loc) == 1, f'Location of {report} plot should have `len`==1, passed `len`={len(loc)}'
                assert col_generation_deals is not None, f'To plot GINI by generations you need to pass `col_generation_deals`'
                ax = plt.subplot2grid((height, width), **loc[0])
                ax_twinx = ax.twinx()
                for col_target in cols_target:
                    self._plot_graph(df,
                                     col_score='',
                                     col_target=col_target,
                                     ax=ax,
                                     report=report,
                                     title=report,
                                     ax_twinx=ax_twinx,
                                     col_generation_deals=col_generation_deals,
                                     name=col_target,
                                     label='Mean Target'
                                     )


            elif report == 'MeanTarget-by-Probability':
                loc = self._get_loc(locations, report)
                assert len(loc) == len(cols_score), f'Location of {report} plot should have `len`=={len(cols_score)}, passed `len`={len(loc)}'
                for loc_score, col_score in zip(loc, cols_score):
                    mask_single_score = ~(df[cols_score + cols_target].isnull().any(axis=1))
                    log.info(f"{report} plot for `{col_score}` uses DataFrame with score. DataFrame's shape = {df[mask_single_score].shape}")
                    ax = plt.subplot2grid((height, width), **loc_score)
                    ax_twinx = ax.twinx()
                    for col_target in cols_target:
                        self._plot_graph(df[mask_single_score],
                                         col_score,
                                         col_target,
                                         ax=ax,
                                         report=report,
                                         ax_twinx=ax_twinx,
                                         title=report+' `'+col_score+'`',
                                         name=col_target)

            elif report == 'Probability-Distribution':
                loc = self._get_loc(locations, report)
                assert len(loc) == len(cols_score), f'Location of {report} plot should have `len`=={len(cols_score)}, passed `len`={len(loc)}'
                for loc_score, col_score in zip(loc, cols_score):
                    mask_single_score = ~(df[cols_score + cols_target].isnull().any(axis=1))
                    ax = plt.subplot2grid((height, width), **loc_score)
                    self._plot_graph(df[mask_single_score], col_score, col_target,
                                     report=report,
                                     ax=ax,
                                     title=report+' `'+col_score+'`',
                                     name=col_target)

            else:
                assert True==False, f'There is no `{report}` option in `plot_report`. You can use `{name_map.values()}`'

    @classmethod
    def _fit_plot(cls, df, col_score, col_target, report, **kwargs):
        try:
            plot_class = cls._class_map[report]
            return plot_class(col_score, col_target, **kwargs).fit(df)
        except KeyError:
            assert True==False, f'There is no `{report}` option in `ReportBinary` module. You can use `{self._class_map.keys()}`'

    @classmethod
    def _plot_graph_class(cls, df, col_score, col_target, report, ax=None, title=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if title is None:
            title=report

        ax_twinx = ax.twinx()

        report_object = cls._fit_plot(df, col_score, col_target, report, ax_twinx = ax_twinx, **kwargs)
        report_object.plot(ax, title=title, ax_twinx = ax_twinx, **kwargs)
        return report_object

    def _plot_graph(self, df, col_score, col_target, report, ax=None, title=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if title is None:
            title=report

        try:
            self._stats[(report, col_score, col_target)].plot(ax, title=title, **kwargs)
        except KeyError:
            self._stats[(report, col_score, col_target)] = self._fit_plot(df, col_score, col_target, report, **kwargs)
            self._stats[(report, col_score, col_target)].plot(ax, title=title, **kwargs)
