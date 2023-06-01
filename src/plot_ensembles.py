""" Author: Indiana
    Date: 14 Dec 2022

    Generate plots of ensembles. (Plots are not saved.)
"""

import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statistics import mean


def main(show_convergence):
    """ Load demographic ensembles
    """
    # California
    # cut edges
    with open('../ensembles/demographic/cut_edges/cutedges_california_1.pkl', 'rb') as f:    tract_cutedges_california_1 = pkl.load(f)
    with open('../ensembles/demographic/cut_edges/cutedges_california_2.pkl', 'rb') as f:    tract_cutedges_california_2 = pkl.load(f)
    with open('../ensembles/demographic/cut_edges/cutedges_california_3.pkl', 'rb') as f:    tract_cutedges_california_3 = pkl.load(f)
    # majority-Hispanic or -Latino
    with open('../ensembles/demographic/majority-minority/majmin_california_1.pkl', 'rb') as f:    majmin_california_1 = pkl.load(f)

    # Cal 3
    # cut edges
    with open('../ensembles/demographic/cut_edges/cutedges_cal3_1.pkl', 'rb') as f:  tract_cutedges_cal3_1 = pkl.load(f)
    with open('../ensembles/demographic/cut_edges/cutedges_cal3_2.pkl', 'rb') as f:  tract_cutedges_cal3_2 = pkl.load(f)
    # majority-Hispanic or -Latino
    with open('../ensembles/demographic/majority-minority/majmin_cal3_1.pkl', 'rb') as f:    majmin_cal3_1 = pkl.load(f)


    """ Load voting ensembles
    """
    # California
    # cut edges
    with open('../ensembles/voting/cut_edges/cutedges_california_1.pkl', 'rb') as f:    precinct_cutedges_california_1 = pkl.load(f)
    with open('../ensembles/voting/cut_edges/cutedges_california_2.pkl', 'rb') as f:    precinct_cutedges_california_2 = pkl.load(f)
    # republican seats
    with open('../ensembles/voting/republican_seats/republican_seats_california_1.pkl', 'rb') as f: republican_seats_california_1 = pkl.load(f)
    # efficiency gap
    with open('../ensembles/voting/efficiency_gap/efficiency_gap_california_1.pkl', 'rb') as f: efficiency_gap_california_1 = pkl.load(f)

    # Cal 3
    # republican seats
    with open('../ensembles/voting/republican_seats/republican_seats_cal3_1.pkl', 'rb') as f:   republican_seats_cal3_1 = pkl.load(f)
    # efficiency gap
    with open('../ensembles/voting/efficiency_gap/efficiency_gap_cal3_1.pkl', 'rb') as f:   efficiency_gap_cal3_1 = pkl.load(f)


    """ Define plot styles
    """
    colors_california = ['tab:cyan', 'tab:orange', 'tab:green']
    colors_cal3 = ['tab:red', 'tab:pink']
    convergence_intervals = [100, 25000, 50000]
    alpha = 0.7
    linewidth = 1.5
    linestyle = 'dashed'
    loc = 'upper right'


    """ Plot tract-level convergence (demographic ensembles)
    """
    if show_convergence:
        # California
        bins = 25
        for steps in convergence_intervals:
            _, ax = plt.subplots()
            plt.title(f'Distribution of # Cut Edges in California after {steps} steps using Tract-level dual graphs')
            plt.hist(tract_cutedges_california_3[:steps], color=colors_california[2], alpha=alpha, bins=bins)
            plt.hist(tract_cutedges_california_2[:steps], color=colors_california[1], alpha=alpha, bins=bins)
            plt.hist(tract_cutedges_california_1[:steps], color=colors_california[0], alpha=alpha, bins=bins)
            plt.axvline(mean(tract_cutedges_california_3[:steps]), color=colors_california[2], linewidth=linewidth, linestyle=linestyle)
            plt.axvline(mean(tract_cutedges_california_2[:steps]), color=colors_california[1], linewidth=linewidth, linestyle=linestyle)
            plt.axvline(mean(tract_cutedges_california_1[:steps]), color=colors_california[0], linewidth=linewidth, linestyle=linestyle)
            seed0 = mpatches.Patch(color=colors_california[0], label='Seed 0')
            seed3 = mpatches.Patch(color=colors_california[1], label='Seed 3')
            seed4 = mpatches.Patch(color=colors_california[2], label='Seed 4')
            ax.legend(
                handles=[seed0, seed3, seed4], 
                loc=loc
            )
            plt.ylabel('# Plans')
            plt.xlabel('# Cut Edges')
            plt.show()
        
        # Cal 3
        bins = 25
        for steps in convergence_intervals:
            _, ax = plt.subplots()
            plt.title(f'Distribution of # Cut Edges in Cal 3 after {steps} steps using Tract-level dual graphs')
            plt.hist(tract_cutedges_cal3_2[:steps], color=colors_cal3[1], alpha=alpha, bins=bins)
            plt.hist(tract_cutedges_cal3_1[:steps], color=colors_cal3[0], alpha=alpha, bins=bins)
            plt.axvline(mean(tract_cutedges_cal3_2[:steps]), color=colors_cal3[1], linewidth=linewidth, linestyle=linestyle)
            plt.axvline(mean(tract_cutedges_cal3_1[:steps]), color=colors_cal3[0], linewidth=linewidth, linestyle=linestyle)
            seed0 = mpatches.Patch(color=colors_cal3[0], label='Seed 0')
            seed1 = mpatches.Patch(color=colors_cal3[1], label='Seed 1')
            ax.legend(
                handles=[seed0, seed1], 
                loc=loc
            )
            plt.ylabel('# Plans')
            plt.xlabel('# Cut Edges')
            plt.show()


    """ Compare California and Cal 3 ensembles of the percentage of majority-Hispanic or -Latino districts
    """
    bins = 13
    _, ax = plt.subplots()
    plt.title('Distribution of % Districts Majority-Hispanic or -Latino in Cal 3 vs. California')
    plt.hist([val/52 for val in majmin_california_1], color=colors_california[0], alpha=alpha, bins=bins)
    plt.hist([val/52 for val in majmin_cal3_1], color=colors_cal3[0], alpha=alpha, bins=bins)
    plt.axvline(mean([val/52 for val in majmin_california_1]), color=colors_california[0], linewidth=linewidth, linestyle=linestyle)
    plt.axvline(mean([val/52 for val in majmin_cal3_1]), color=colors_cal3[0], linewidth=linewidth, linestyle=linestyle)
    plt.axvline(.394, color='darkorange', linewidth=linewidth, linestyle=(0, (5, 1)))
    patch1 = mpatches.Patch(color=colors_california[0], label='California (seed 0)')
    patch2 = mpatches.Patch(color=colors_cal3[0], label='Cal 3 (seed 0)')
    patch3 = mpatches.Patch(color='darkorange', label='% population (actual)')
    ax.legend(
        handles=[patch1, patch2, patch3], 
        loc=loc
    )
    plt.ylabel('# Plans')
    plt.xlabel('% Districts Majority-Hispanic or -Latino')
    plt.show()


    """ Plot precinct-level convergence (voting ensembles)
    """
    if show_convergence:
        # California 
        bins = 25
        for steps in convergence_intervals:
            _, ax = plt.subplots()
            plt.title(f'Distribution of # Cut Edges in California after {steps} steps using Precinct-level dual graphs')
            plt.hist(precinct_cutedges_california_2[:steps], color=colors_california[1], alpha=alpha, bins=bins)
            plt.hist(precinct_cutedges_california_1[:steps], color=colors_california[0], alpha=alpha, bins=bins)
            plt.axvline(mean(precinct_cutedges_california_2[:steps]), color=colors_california[1], linewidth=linewidth, linestyle=linestyle)
            plt.axvline(mean(precinct_cutedges_california_1[:steps]), color=colors_california[0], linewidth=linewidth, linestyle=linestyle)
            seed0 = mpatches.Patch(color=colors_california[0], label='Seed 0')
            seed3 = mpatches.Patch(color=colors_california[1], label='Seed 3')
            ax.legend(
                handles=[seed0, seed3], 
                loc=loc
            )
            plt.ylabel('# Plans')
            plt.xlabel('# Cut Edges')
            plt.show()


    """ Compare California and Cal 3 ensembles of the percentage of Republican seats
    """
    bins = 8
    _, ax = plt.subplots()
    plt.title('Distribution of Republican Seat Share in Cal 3 vs. California')
    plt.hist([val/52 for val in republican_seats_california_1], color=colors_california[0], alpha=alpha, bins=bins)
    plt.hist([val/52 for val in republican_seats_cal3_1], color=colors_cal3[0], alpha=alpha, bins=bins)
    plt.axvline(mean([val/52 for val in republican_seats_california_1]), color=colors_california[0], linewidth=linewidth, linestyle=linestyle)
    plt.axvline(mean([val/52 for val in republican_seats_cal3_1]), color=colors_cal3[0], linewidth=linewidth, linestyle=linestyle)
    plt.axvline(.3191, color='darkorange', linewidth=linewidth, linestyle=(0, (5, 1)))
    patch1 = mpatches.Patch(color=colors_california[0], label='California (seed 0)')
    patch2 = mpatches.Patch(color=colors_cal3[0], label='Cal 3 (seed 0)')
    patch3 = mpatches.Patch(color='darkorange', label='Vote share (actual)')
    ax.legend(
        handles=[patch1, patch2, patch3], 
        loc=loc
    )
    plt.ylabel('# Plans')
    plt.xlabel('Republican Seat Share')
    plt.show()


    """ Compare California and Cal 3 ensembles of efficiency gap favoring Republicans
    """
    bins = 25
    _, ax = plt.subplots()
    plt.title('Distribution of the Efficiency Gap of Cal 3 vs. California')
    plt.hist(efficiency_gap_california_1, color=colors_california[0], alpha=alpha, bins=bins)
    plt.hist(efficiency_gap_cal3_1, color=colors_cal3[0], alpha=alpha, bins=bins)
    plt.axvline(mean(efficiency_gap_california_1), color=colors_california[0], linewidth=linewidth, linestyle=linestyle)
    plt.axvline(mean(efficiency_gap_cal3_1), color=colors_cal3[0], linewidth=linewidth, linestyle=linestyle)
    patch1 = mpatches.Patch(color=colors_california[0], label='California (seed 0)')
    patch2 = mpatches.Patch(color=colors_cal3[0], label='Cal 3 (seed 0)')
    ax.legend(
        handles=[patch1, patch2], 
        loc=loc
    )
    plt.ylabel('# Plans')
    plt.xlabel('Efficiency Gap')
    plt.show()
        

if __name__ == '__main__':
    show_convergence = True
    main(show_convergence)