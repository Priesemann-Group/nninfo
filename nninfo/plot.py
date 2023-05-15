from math import comb

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import nninfo

_dataset_display_names = {
    'full_set/train': 'Train Set', 'full_set/test': 'Test Set'}


##### Performance plots #####

def plot_accuracy(performance, dataset_name, ax, **kwargs):

    # Get maximum epoch and format broken axis
    max_epoch_exponent = int(np.ceil(np.log10(performance['epoch_id'].max())))
    nninfo.plot.format_figure_broken_axis(ax, max_exp=max_epoch_exponent)

    dataset_display_name = _dataset_display_names.get(
        dataset_name, dataset_name)
    
    kwargs = kwargs.copy()
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('label', f'{dataset_display_name} Accuracy')
    
    # Plot loss
    nninfo.plot.plot_mean_and_interval(performance, ax, x=('epoch_id', ''), y=(
        dataset_name, 'accuracy'), **kwargs)


def plot_loss(performance, dataset_name, ax, **kwargs):

    # Get maximum epoch and format broken axis
    max_epoch_exponent = int(np.ceil(np.log10(performance['epoch_id'].max())))
    nninfo.plot.format_figure_broken_axis(ax, max_exp=max_epoch_exponent)

    dataset_display_name = _dataset_display_names.get(
        dataset_name, dataset_name)

    kwargs = kwargs.copy()
    kwargs.setdefault('lw', 1)
    kwargs.setdefault('label', f'{dataset_display_name} Accuracy')

    # Plot loss
    nninfo.plot.plot_mean_and_interval(performance, ax, x=('epoch_id', ''), y=(
        dataset_name, 'loss'), **kwargs)


def plot_loss_accuracy(performance, ax, ax2, dataset_names=['full_set/train', 'full_set/test'], **kwargs):
    """
    Plot loss and accuracy.
    """

    max_epoch_exponent = int(np.ceil(np.log10(performance['epoch_id'].max())))
    nninfo.plot.format_figure_broken_axis(ax, max_exp=max_epoch_exponent)

    kwargs = kwargs.copy()
    kwargs.setdefault('lw', 1)

    for dataset_name in dataset_names:

        dataset_display_name = _dataset_display_names.get(
            dataset_name, dataset_name)

        # Plot accuracy
        nninfo.plot.plot_mean_and_interval(performance, ax, x=('epoch_id', ''), y=(
            dataset_name, 'accuracy'), label=f'{dataset_display_name} Accuracy', **kwargs)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.plot([], label=f'{dataset_display_name} Loss')

        # Plot loss
        ax2.plot([])  # advance color cycle
        nninfo.plot.plot_mean_and_interval(performance, ax2, x=(
            'epoch_id', ''), y=(dataset_name, 'loss'), **kwargs)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")

        max_epoch = performance['epoch_id'].max()

        print(f'Avg. {dataset_display_name} accuracy at epoch {max_epoch}:',
              performance[performance['epoch_id'] ==
                          max_epoch][(dataset_name, 'accuracy')].mean(),
              '+-',
              performance[performance['epoch_id'] == max_epoch][(dataset_name, 'accuracy')].std(ddof=1))

###### PID PLOTS ######


def plot_representational_complexity(pid_summary: pd.DataFrame, ax: plt.Axes, quantile_level=0.95, use_median=False, **kwargs):

    max_epoch_exponent = int(np.ceil(np.log10(pid_summary['epoch_id'].max())))
    format_figure_broken_axis(ax, max_epoch_exponent)

    plot_mean_and_interval(
        pid_summary, ax, x='epoch_id', y='representational_complexity', quantile_level=quantile_level, use_median=use_median, **kwargs)


def plot_degree_of_synergy_atoms(pid_summary: pd.DataFrame, ax: plt.Axes, quantile_level=0.95, use_median=False, **kwargs):

    max_epoch_exponent = int(np.ceil(np.log10(pid_summary['epoch_id'].max())))
    format_figure_broken_axis(ax, max_epoch_exponent)

    max_degree = pid_summary['degree_of_synergy_atoms'].columns.max()

    for degree in range(1, max_degree + 1):
        plot_mean_and_interval(
            pid_summary, ax, x=pid_summary['epoch_id'], y=('degree_of_synergy_atoms', degree), quantile_level=quantile_level, use_median=use_median, label=f'Deg. of syn. {degree}', **kwargs)

def plot_reing_directed_differences(reing_results: pd.DataFrame, ax: plt.Axes, quantile_level=0.95, use_median=False, labels=None, **kwargs):

    max_epoch_exponent = int(np.ceil(np.log10(reing_results['epoch_id'].max())))
    format_figure_broken_axis(ax, max_epoch_exponent)

    # Find all columns that match the patterh 'C(k||k+1)'
    for reing_column_name in reing_results.columns[reing_results.columns.str.match('C\(\d+\|\|\d+\)')]:
        label = reing_column_name if labels is None else labels[reing_column_name]
        plot_mean_and_interval(
            reing_results, ax, x=reing_results['epoch_id'], y=reing_column_name, quantile_level=quantile_level, use_median=use_median, label=reing_column_name, **kwargs)

def plot_reing_complexity(reing_results: pd.DataFrame, ax: plt.Axes, quantile_level=0.95, use_median=False, **kwargs):

    max_epoch_exponent = int(np.ceil(np.log10(reing_results['epoch_id'].max())))
    format_figure_broken_axis(ax, max_epoch_exponent)

    plot_mean_and_interval(
        reing_results, ax, x=reing_results['epoch_id'], y='reing_complexity', quantile_level=quantile_level, use_median=use_median, **kwargs)

##### HELPER FUNCTIONS ######

def plot_mean_and_interval(df, ax, x=('epoch_id', ''), y='c', use_median=False, quantile_level=0.95, zorder_shift=0, **kwargs):
    df_center = df.groupby(x).median(
    ) if use_median else df.groupby(df.epoch_id).mean()
    df_high = df.groupby(df.epoch_id).quantile(.5 - quantile_level / 2)
    df_low = df.groupby(df.epoch_id).quantile(.5 + quantile_level / 2)

    kwargs.setdefault('label', '')

    line = ax.plot(df_center[y], zorder=1+zorder_shift,
                   solid_capstyle='butt', **kwargs)
    ax.fill_between(df_low.index, df_low[y].values,
                    df_high[y].values, color=line[-1].get_color(), alpha=0.3, zorder=0+zorder_shift, linewidth=0)


def format_figure_broken_axis(ax, max_exp=4):

    ax.set_xscale('symlog', linthresh=1, linscale=.6)

    ax.set_xlim(0, 10**max_exp)
    ax.set_xticks([0] + [10**i for i in range(max_exp+1)])
    ax.set_xticklabels(['$0$', '$1$'] + ['' if i %
                                         2 == 1 else f'$10^{i}$' for i in range(1, max_exp+1)])

    # Broken axis
    d = .01
    broken_x = 0.07
    breakspacing = 0.015
    ax.plot((broken_x-breakspacing*0.9, broken_x+breakspacing*0.9), (0, 0),
            color='w', transform=ax.transAxes, clip_on=False, linewidth=.8, zorder=3)
    ax.plot((broken_x-breakspacing*0.9, broken_x+breakspacing*0.9), (1, 1),
            color='w', transform=ax.transAxes, clip_on=False, linewidth=.8, zorder=3)

    kwargs = dict(transform=ax.transAxes, color='k',
                  clip_on=False, linewidth=.8, zorder=4)
    ax.plot((broken_x-d-breakspacing, broken_x+d -
             breakspacing), (-3*d, +3*d), **kwargs)
    ax.plot((broken_x-d-breakspacing, broken_x+d -
             breakspacing), (1-3*d, 1+3*d), **kwargs)
    ax.plot((broken_x-d+breakspacing, broken_x+d +
             breakspacing), (-3*d, +3*d), **kwargs)
    ax.plot((broken_x-d+breakspacing, broken_x+d +
             breakspacing), (1-3*d, 1+3*d), **kwargs)