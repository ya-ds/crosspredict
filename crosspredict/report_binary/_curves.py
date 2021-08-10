from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import logging
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay


class ByMonthsCurve:
    def _draw_series_to_bar_plot(self, df_series, ax, label='Count'):
        N = len(df_series.index.values)
        ind = np.arange(N)

        # ax.bar(ind, df_series.values, 0.5, color='r')
        g = sns.barplot(x=df_series.index, y=df_series, ax=ax, palette="BuPu")

        ax.set_ylabel(label, color='r')
        ax.tick_params('y', colors='r')

        plt.sca(ax)
        if df_series.index.dtype == '<M8[ns]':
            plt.xticks(ind, df_series.index.map(
                lambda x: x.strftime("%Y-%m")).sort_values(ascending=True))
        else:
            plt.xticks(ind, df_series.index.values)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

    def _draw_series_to_line_plot(self, df_series, ax, label, name, legend=False, bottom=0, auto=True, ax_twinx=None, col_generation_deals=None):
        N = len(df_series.index.values)
        ind = np.arange(N)
        ax.plot(ind, df_series.values, label=name)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=bottom, auto=auto)
        ax.tick_params('y')
        if legend:
            ax.legend(loc="lower right")


class CurveFabric(ABC, ByMonthsCurve):
    def __init__(self, col_score, col_target, name=None):
        self.col_score = col_score
        self.col_target = col_target
        if name:
            self.name = name
        else:
            self.name = self.col_score + ' | ' + self.col_target

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def plot(self, ax=None, ax_twinx=None, title=None, label=None, **kwargs):
        pass


class DistributionCurve(CurveFabric):
    def __init__(self, col_score, col_target, name=None, **kwargs):
        super().__init__(col_score, col_target, name=name)
        self.series = None

    def fit(self, df):
        log = logging.getLogger(__name__)
        log.info(f"Fitting DistributionCurve.")
        assert sum(df[self.col_score].isnull())==0, f"Column `{self.col_score}` passed in `DistributionCurve` contains None objects"
        log.info(f"Using DataFrame where {self.col_score} is not None. DataFrame's shape = {df.shape}")
        self.series = df[self.col_score]
        return self

    def plot(self, ax=None, title=None, **kwargs):
        self.ax = ax
        sns.histplot(self.series, ax=ax)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        return self


class GenGINICurve(CurveFabric):
    def __init__(self, col_score, col_target, col_generation_deals, name=None, **kwargs):
        super().__init__(col_score, col_target, name=name)
        self.series_auc = None
        self._col_generation_deals = col_generation_deals
        self.count = None

    def fit(self, df):
        self.count = df.groupby(self._col_generation_deals).size()
        self.series_auc = df[~df[self.col_score].isnull()].groupby(self._col_generation_deals).apply(
            lambda x: 100 * (2 * roc_auc_score(x[self.col_target], x[self.col_score]) - 1) if len(
                x[self.col_target].unique()) > 1 else None)
        self.series_auc = self.series_auc.reindex(index=self.count.index)
        return self

    def plot(self, ax=None, title=None, **kwargs):
        ax_twinx = kwargs['ax_twinx']
        self.ax = ax
        self.ax_twinx = ax_twinx
        self._draw_series_to_bar_plot(self.count, ax, label='Count')
        self._draw_series_to_line_plot(self.series_auc, ax=ax_twinx, name=self.name, label='GINI', legend=True, **kwargs)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        return self


class GenRiskCurve(CurveFabric):
    def __init__(self, col_score, col_target, col_generation_deals, name=None, **kwargs):
        super().__init__(col_score, col_target, name=name)
        self._col_generation_deals = col_generation_deals
        self.risk = None

    def fit(self, df):
        self.risk = df.groupby(self._col_generation_deals)[self.col_target].agg(['count', 'mean'])
        return self

    def plot(self, ax=None, title=None, label=None, name=None, **kwargs):
        ax_twinx = kwargs['ax_twinx']
        self.ax = ax
        self.ax_twinx = ax_twinx
        if label is None:
            label='Mean Target'
        self._draw_series_to_bar_plot(self.risk['count'], ax, label='Count')
        self._draw_series_to_line_plot(self.risk['mean'], ax_twinx, label=label, name=name, legend=True)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        return self


class MeanProbCurve(CurveFabric):
    def __init__(self, col_score, col_target, name=None, **kwargs):
        super().__init__(col_score, col_target, name=name)
        self.pd_deciles = None
        self.pd_bins = None
        self.mean_prob = None

    def fit(self, df):
        self.pd_deciles, self.pd_bins = pd.qcut(df[self.col_score], q=np.arange(
            0, 1.000001, 0.1), duplicates='drop', retbins=True)
        self.mean_prob = df.groupby(self.pd_deciles)[
            self.col_target].agg(['count', 'mean'])
        return self

    def plot(self, ax=None, ax_twinx=None, title=None, label=None, name=None, **kwargs):
        self.ax = ax
        self.ax_twinx = ax_twinx
        if label is None:
            label='Mean Target'
        self._draw_series_to_bar_plot(self.mean_prob['count'], ax, label='Count')
        self._draw_series_to_line_plot(self.mean_prob['mean'], ax_twinx, label=label, name=name, legend=True)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        return self


class RocAucCurve(CurveFabric):
    def __init__(self, col_score, col_target, name=None, **kwargs):
        super().__init__(col_score, col_target, name=name)
        self.fpr = None
        self.tpr = None
        self.roc_auc = None

    def fit(self, df):
        self.fpr, self.tpr, _ = roc_curve(
            df[self.col_target], df[self.col_score])
        self.roc_auc = auc(self.fpr, self.tpr)
        return self

    def plot(self, ax=None, title=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax

        self.viz = RocCurveDisplay(
            fpr=self.fpr,
            tpr=self.tpr,
            roc_auc=self.roc_auc*100,
            estimator_name=self.name)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')

        self.viz.plot(ax=ax, name=self.name, **kwargs)
        return self


class PrecisionRecallCurve(CurveFabric):
    def __init__(self, col_score, col_target, name=None, **kwargs):
        super().__init__(col_score, col_target, name=name)
        self.precision = None
        self.recall = None
        self.average_precision = None

    def fit(self, df):
        self.precision, self.recall, _ = precision_recall_curve(
            df[self.col_target], df[self.col_score])
        self.average_precision = average_precision_score(
            df[self.col_target], df[self.col_score])
        return self

    def plot(self, ax=None, title=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()
        self.ax = ax

        self.viz = PrecisionRecallDisplay(
            precision=self.precision, recall=self.recall,
            average_precision=self.average_precision, estimator_name=self.name
        )

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')

        self.viz.plot(ax=ax, name=self.name, **kwargs)
        return self
