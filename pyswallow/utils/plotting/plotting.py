import matplotlib.pyplot as plt
import numpy as np

from .plot_designer import PlotDesigner
from ..history import SOHistory
from ...handlers.archive import Archive

from typing import Union


def plot_fitness_history(history: SOHistory,
                         title: str,
                         designer: Union[PlotDesigner, None] = None,
                         save: Union[bool, None] = None) -> None:

    """Generates a plot of the optimisations fitness / iterations.

    Parameters
    ----------
    history : SOHistory
        Contains all the information used to generate the plots.
    title : str
        Title to be placed on the plot.
    designer : PlotDesigner
        Contains information required to format the plot.
    save : bool
        Whether to save the plot or not.
    """

    if not issubclass(type(history), SOHistory):
        raise TypeError('history must be a class of SOHistory.')

    n_iterations = len(history.arr_mean_fitness)

    if designer is None:
        designer = PlotDesigner()
        designer.label = ['Iterations', 'Fitness']

    fig, ax = plt.subplots(1, 1, figsize=designer.figsize)

    ax.scatter(np.arange(n_iterations), history.arr_best_fitness, label='Best')
    ax.scatter(np.arange(n_iterations), history.arr_mean_fitness, label='Mean')

    ax.set_title(title, fontsize=designer.title_fontsize)
    ax.legend(fontsize=designer.text_fontsize)

    ax.set_xlabel(designer.label[0])
    ax.set_ylabel(designer.label[1])

    ax.tick_params(labelsize=designer.text_fontsize)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()


def plot_archive(archive: Archive,
                 title: str,
                 designer: Union[PlotDesigner, None] = None,
                 save: Union[bool, None] = None) -> None:

    """Generates a plot of the Pareto front.

    Parameters
    ----------
    archive : Archive
        Population from which to generate the plots.
    title : str
        Title of the plot to produce.
    designer : PlotDesigner
        Contains plot formatting information.
    save : str or None
        File path to save the plot to.
    """

    if not issubclass(type(archive), Archive):
        raise TypeError('archive must be a class of Archive.')

    n_obj = len(archive.population[0].fitness)
    if not n_obj == 2:
        raise ValueError('can only show Pareto front of 2 objectives.')

    if designer is None:
        designer = PlotDesigner()
        designer.label = ['f1', 'f2']

    fig, ax = plt.subplots(1, 1, figsize=designer.figsize)

    f1 = [swallow.fitness[0] for swallow in archive.population]
    f2 = [swallow.fitness[1] for swallow in archive.population]

    ax.scatter(f1, f2, label='Pareto Front')

    ax.set_title(title, fontsize=designer.title_fontsize)
    ax.legend(fontsize=designer.text_fontsize)

    ax.set_xlabel(designer.label[0])
    ax.set_ylabel(designer.label[1])

    ax.tick_params(labelsize=designer.text_fontsize)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()
