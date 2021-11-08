import math
from typing import Any
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def plot_diagonal(ax: Axes, **kwargs: Any):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    limits = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.plot(limits, limits, **kwargs)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def plot_regression(y_true,
                    y_pred,
                    kind: str = 'hex',
                    height: int = 7,
                    include_statistics: bool = True) -> Figure:
    if kind == 'hex':
        joint_kws = {'gridsize': 20}
    else:
        joint_kws = {}

    g = sns.jointplot(x=y_true, y=y_pred, kind=kind,
                      joint_kws=joint_kws, height=height)
    g.plot_joint(sns.regplot, scatter=False)
    plot_diagonal(g.ax_joint, ls="--", c='steelblue')

    g.set_axis_labels('True Values', 'Predictions')

    if include_statistics:
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        text = f'RMSE = {rmse:.3f}\n$R^2$ = {r2:.2f}'
        g.ax_joint.text(0.05, 0.9, text, transform=g.ax_joint.transAxes)

    return g.fig
