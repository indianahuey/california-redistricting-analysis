""" Author: Indiana
    Date: 13 Dec 2022

    Generate ensembles and pickle them into ../ensembles/.
"""

import timeit
import pickle as pkl
import pandas as pd
import geopandas as gpd
import networkx as nx
from gerrychain.random import random
from gerrychain import Graph, Partition, constraints, MarkovChain
from gerrychain.updaters import cut_edges, Tally
from gerrychain.tree import recursive_tree_part
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
from functools import partial


def __random_partitions(graph, num_districts, idealpop, totpop_key='total_pop', pop_tolerance=0.02, updaters=None, seeds=[]):
    """ Return random partitions of a graph.
    """
    # make random plans
    plans = []
    for seed in seeds:
        random.seed(seed)
        plans.append(
            recursive_tree_part(
                graph,
                range(num_districts),
                idealpop,
                totpop_key,
                pop_tolerance,
                10
            )
        )

    # return partitions from plans
    return [
        Partition(
            graph,
            plan,
            updaters
        ) for plan in plans
    ]


def make_demographic_ensembles(steps, make_objects=False):
    """ Generate and pickle tract-level ensembles of the
        number of majority-Hispanic or -Latino districts.
        
        Ensembles of the number of cut edges
        are recorded as well. 
    """
    print('Generating demographic ensembles...')


    if make_objects:
        """ Process data
        """
        # read data
        tracts = gpd.read_file('../data/demographic/tl_2022_06_tract/tl_2022_06_tract.shp')
        demographics = pd.read_csv('../data/demographic/DECENNIALPL2020.P2_2022-12-13T214431/DECENNIALPL2020.P2-Data.csv', skiprows=1)

        # pare down and format data
        demographics = demographics[['Geography', ' !!Total:', ' !!Total:!!Hispanic or Latino']]
        demographics = demographics.rename(columns={
            'Geography': 'GEOID',
            ' !!Total:': 'total_pop',
            ' !!Total:!!Hispanic or Latino': 'hispanic_latino_pop'
        })
        demographics['GEOID'] = demographics['GEOID'].str[9:]

        # merge data
        data = tracts.merge(demographics, on='GEOID', how='left')


    """ Create dual graphs
    """
    # create dual graph of California
    if make_objects:
        graph_california = Graph.from_geodataframe(data)
        with open('../objects/graphs/demographic/california_graph.pkl', 'wb') as f: pkl.dump(graph_california, f)
    else:
        with open('../objects/graphs/demographic/california_graph.pkl', 'rb') as f: graph_california = pkl.load(f)

    # define NorCal, Cal, SoCal by county fips
    # names: https://sf.curbed.com/2018/6/14/17464134/three-californias-tim-draper-ballot-iniative
    # fips: https://www.weather.gov/hnx/cafips
    # San Benito, Monterey, San Luis Obispo, Santa Barbara, Ventura, Los Angeles
    fips_cal = {'069', '053', '079', '083', '111', '037'}
    # Mono, Madera, Fresno, Kings, Tulare, Inyo, Kern, San Bernardino, Riverside, Orange, San Diego, Imperial
    fips_socal = {'051', '039', '019', '107', '027', '029', '071', '065', '059', '073', '025'}
    # everything else
    if make_objects:
        fips_norcal = set(
            [ 
                fp for fp in data['COUNTYFP']
                if fp not in fips_cal and fp not in fips_socal
            ]
        )
        with open('../objects/fips/demographic_norcal_fips.pkl', 'wb') as f:    pkl.dump(fips_norcal, f)
    else:
        with open('../objects/fips/demographic_norcal_fips.pkl', 'rb') as f:    fips_norcal = pkl.load(f)

    # create dual graphs of NorCal, Cal, SoCal
    if make_objects:
        graph_norcal, graph_cal, graph_socal = [
            subgraph for subgraph in [
                nx.subgraph(
                    graph_california,
                    [
                        node for node in graph_california.nodes()
                        if graph_california.nodes()[node]['COUNTYFP'] in fips
                    ]
                ) for fips in [fips_norcal, fips_cal, fips_socal]
            ]
        ]
        with open('../objects/graphs/demographic/graph_norcal.pkl', 'wb') as f: pkl.dump(graph_norcal, f)
        with open('../objects/graphs/demographic/graph_cal.pkl', 'wb') as f: pkl.dump(graph_cal, f)
        with open('../objects/graphs/demographic/graph_socal.pkl', 'wb') as f: pkl.dump(graph_socal, f)
    else:
        with open('../objects/graphs/demographic/graph_norcal.pkl', 'rb') as f: graph_norcal = pkl.load(f)
        with open('../objects/graphs/demographic/graph_cal.pkl', 'rb') as f: graph_cal = pkl.load(f)
        with open('../objects/graphs/demographic/graph_socal.pkl', 'rb') as f: graph_socal = pkl.load(f)

    """ Define population variables
    """
    # number of seats for California
    num_districts_california = 52

    # total population
    totpop_california, totpop_norcal, totpop_cal, totpop_socal = [
        sum(
            [
                graph.nodes()[node]['total_pop']
                for node in graph.nodes()
            ]
        ) for graph in [graph_california, graph_norcal, graph_cal, graph_socal]
    ]

    # ideal population for California
    idealpop_california = totpop_california / num_districts_california

    # number of seats for NorCal, Cal, SoCal
    num_districts_norcal, num_districts_cal, num_districts_socal = [
        round(num_districts_california * (totpop / totpop_california))
        for totpop in [totpop_norcal, totpop_cal, totpop_socal] 
    ]
    # make sure no total seats are lost or gained due to rounding
    if sum([num_districts_norcal, num_districts_cal, num_districts_socal]) != num_districts_california:
        raise Exception("Total number of seats across Cal 3 doesn't match total number of seats for California.")

    # ideal population for NorCal, Cal, SoCal
    idealpop_norcal, idealpop_cal, idealpop_socal = [
        totpop / num_districts
        for totpop, num_districts in zip(
            [totpop_norcal, totpop_cal, totpop_socal],
            [num_districts_norcal, num_districts_cal, num_districts_socal]
        )
    ]


    """ Initialize partitions 
    """
    # define updaters
    my_updaters = {
        'cut_edges': cut_edges,
        'district_totpop': Tally('total_pop', alias='district_totpop'),
        'district_hispanic_latino_pop': Tally('hispanic_latino_pop', alias='district_hispanic_latino_pop')
    }

    # create three random partitions for California
    part_california_1, part_california_2, part_california_3 = __random_partitions(
        graph_california,
        num_districts_california,
        idealpop_california,
        updaters=my_updaters,
        seeds=[0, 3, 4]
    )

    # create two random partitions for NorCal, Cal, SoCal each
    part_norcal_1, part_norcal_2 = __random_partitions(
        graph_norcal,
        num_districts_norcal,
        idealpop_norcal,
        updaters=my_updaters,
        seeds=[0, 1]
    )

    part_cal_1, part_cal_2 = __random_partitions(
        graph_cal,
        num_districts_cal,
        idealpop_cal,
        updaters=my_updaters,
        seeds=[0, 1]
    )

    part_socal_1, part_socal_2 = __random_partitions(
        graph_socal,
        num_districts_socal,
        idealpop_socal,
        updaters=my_updaters,
        seeds=[0, 1]
    )


    """ Prepare random walks 
    """
    pop_tolerance = 0.02

    # define proposal methods
    proposal_california, proposal_norcal, proposal_cal, proposal_socal = [
        partial(
            recom,
            pop_col='total_pop',
            pop_target=idealpop,
            epsilon=pop_tolerance,
            node_repeats=1
        ) for idealpop in [idealpop_california, idealpop_norcal, idealpop_cal, idealpop_socal]
    ]

    # define population constraints
    constraint_california, constraint_norcal, constraint_cal, constraint_socal = [
        constraints.within_percent_of_ideal_population(
            partition,
            pop_tolerance,
            pop_key='district_totpop'
        ) for partition in [part_california_1, part_norcal_1, part_cal_1, part_socal_1]
    ]

    # initialize random walks
    def __markov_chain(proposal, constraint, initial_state, steps=steps):
        return MarkovChain(
            proposal=proposal,
            constraints=[constraint],
            accept=always_accept,
            initial_state=initial_state,
            total_steps=steps
        )

    rw_california_1, rw_california_2, rw_california_3 = [
        __markov_chain(
            proposal_california,
            constraint_california,
            partition,
        ) for partition in [part_california_1, part_california_2, part_california_3]
    ]
    rw_norcal_1, rw_norcal_2 = [
        __markov_chain(
            proposal_norcal,
            constraint_norcal,
            partition
        ) for partition in [part_norcal_1, part_norcal_2]
    ]
    rw_cal_1, rw_cal_2 = [
        __markov_chain(
            proposal_cal,
            constraint_cal,
            partition
        ) for partition in [part_cal_1, part_cal_2]
    ]
    rw_socal_1, rw_socal_2 = [
        __markov_chain(
            proposal_socal,
            constraint_socal,
            partition
        ) for partition in [part_socal_1, part_socal_2]
    ]


    """ Do random walks 
    """
    def __walk(rw, num_districts):
        cutedges_ens = []
        majmin_ens = []
        for part in rw:
            count = 0
            for i in range(num_districts):
                # count number of majority-Hispanic or -Latino districts
                if part['district_hispanic_latino_pop'][i] / part['district_totpop'][i] > 0.5:
                    count += 1
            cutedges_ens.append(len(part['cut_edges']))
            majmin_ens.append(count)
        return cutedges_ens, majmin_ens

    print('Walking...', end='\n\t')
    start = timeit.default_timer()

    # ensembles of number of cut edges, ensembles of number of majority-Hispanic or -Latino districts
    # California
    cutedges_california_1, majmin_california_1 = __walk(rw_california_1, num_districts_california)
    print(f'1/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_california_2, majmin_california_2 = __walk(rw_california_2, num_districts_california)
    print(f'2/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_california_3, majmin_california_3 = __walk(rw_california_3, num_districts_california)
    print(f'3/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()

    # NorCal
    cutedges_norcal_1, majmin_norcal_1 = __walk(rw_norcal_1, num_districts_norcal)
    print(f'4/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_norcal_2, majmin_norcal_2 = __walk(rw_norcal_2, num_districts_norcal)
    print(f'5/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()

    # Cal
    cutedges_cal_1, majmin_cal_1 = __walk(rw_cal_1, num_districts_cal)
    print(f'6/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_cal_2, majmin_cal_2 = __walk(rw_cal_2, num_districts_cal)
    print(f'7/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()

    # SoCal
    cutedges_socal_1, majmin_socal_1 = __walk(rw_socal_1, num_districts_socal)
    print(f'8/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_socal_2, majmin_socal_2 = __walk(rw_socal_2, num_districts_socal)
    print(f'9/9, {round((timeit.default_timer() - start) / 60, 2)} min')


    """ Aggregate NorCal, Cal, SoCal ensembles into Cal 3 ensembles
    """
    # number of cut edges
    cutedges_cal3_1 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            cutedges_norcal_1, cutedges_cal_1, cutedges_socal_1
        )
    ]
    cutedges_cal3_2 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            cutedges_norcal_2, cutedges_cal_2, cutedges_socal_2
        )
    ]
    # number of majority-Hispanic or -Latino
    majmin_cal3_1 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            majmin_norcal_1, majmin_cal_1, majmin_socal_1
        )
    ]
    majmin_cal3_2 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            majmin_norcal_2, majmin_cal_2, majmin_socal_2
        )
    ]


    """ Pickle ensembles
    """
    def __pickle(category, object, name):
        with open(f'../ensembles/demographic/{category}/{name}.pkl', 'wb') as f:
            pkl.dump(object, f)

    __pickle('cut_edges', cutedges_california_1, 'cutedges_california_1')
    __pickle('cut_edges', cutedges_california_2, 'cutedges_california_2')
    __pickle('cut_edges', cutedges_california_3, 'cutedges_california_3')

    __pickle('cut_edges', cutedges_cal3_1, 'cutedges_cal3_1')
    __pickle('cut_edges', cutedges_cal3_2, 'cutedges_cal3_2')

    __pickle('majority-minority', majmin_california_1, 'majmin_california_1')
    __pickle('majority-minority', majmin_california_2, 'majmin_california_2')
    __pickle('majority-minority', majmin_california_3, 'majmin_california_3')

    __pickle('majority-minority', majmin_cal3_1, 'majmin_cal3_1')
    __pickle('majority-minority', majmin_cal3_2, 'majmin_cal3_2')


    """ Print information
    """
    print('Info:', end='\n\t')
    print(f'Number of steps = {steps}', end='\n\t')

    # total Hispanic or Latino population for California
    total_hispanic_latino_pop = sum(
        [
            graph_california.nodes()[node]['hispanic_latino_pop']
            for node in graph_california.nodes()
        ]
    )

    print(f'Percent Hispanic or Latino in California: {round((total_hispanic_latino_pop / totpop_california) * 100, 2)}', end='\n\t')

    print(f'Number of seats:', end='\n\t\t')
    print(f'California = {num_districts_california}', end='\n\t\t')
    print(f'NorCal = {num_districts_norcal}', end='\n\t\t')
    print(f'Cal = {num_districts_cal}', end='\n\t\t')
    print(f'SoCal = {num_districts_socal}')

    print('...Generated demographic ensembles.')


def make_voting_ensembles(steps, make_objects=False):
    """ Generate and pickle precinct-level ensembles of the
        number of seats won by Republicans and the
        efficiency gap favoring Republicans of each plan.

        Ensembles of the number of cut edges
        are recorded as well. 

        Note: This method bases population on casted votes, such that
              regional population counts rely on the likely false assumption
              that voter turnout is even across regions. As a consequence,
              regional population counts here likely problematically
              underrepresent some regions more than others.

              This is done as voting data is only available on the precinct level, while
              population data is only available on the Census tract level, and there is no
              shared key by which to join them---except for on the county level, which is
              too large for generating ensembles or making plans in California in general.
    """
    print('Generating voting ensembles...')


    if make_objects:
        """ Process Data 
        """
        # read data
        precincts = gpd.read_file('../data/voting/california-2016-election-precinct-maps-master/california.shp')
        voting_data = pd.read_csv('../data/voting/california-2016-election-precinct-maps-master/all_precinct_results.csv')

        # pare down and format data
        precincts['COUNTYFP'] = precincts['pct16'].str[:3]
        voting_data['total_votes'] = sum(
            [
                voting_data[entry] for entry in
                ['pres_clinton', 'pres_trump', 'pres_johnson', 'pres_stein', 'pres_lariva', 'pres_other']
            ]
        )
        voting_data = voting_data.rename(columns={
            'pres_clinton': 'democrat_votes',
            'pres_trump': 'republican_votes'
        })
        voting_data = voting_data[['pct16', 'total_votes', 'democrat_votes', 'republican_votes']]

        # merge data
        data = precincts.merge(voting_data, on='pct16', how='inner')

        # repair invalid geometries
        invalid_rows = [
            1162, 1164, 1165, 1167, 1173, 1181, 1182, 1184, 1187, 1330, 2483, 2624, 2803, 2841, 2898, 2902, 2909, 2922, 
            2929, 2945, 2949, 2975, 2976, 2985, 2990, 2991, 2995, 2997, 3193, 3304, 3320, 3327, 3447, 3552, 3648, 3715, 
            3857, 3882, 3884, 3886, 3892, 3915, 3937, 3939, 3953, 3999, 4000, 4009, 4013, 5812, 6926, 7122, 7951, 9156, 
            9253, 9299, 9424, 9647, 9650, 9703, 9760, 9858, 9897, 9948, 10092, 10484, 10577, 12090, 12388, 12692, 12697, 
            12699, 13268, 13400, 13447, 13665, 15760, 18636, 18873, 21030, 21032, 21046, 21055, 21074, 21097, 21115, 21128, 
            21161, 21185, 21186, 21798, 22839, 23207, 23429, 23431, 23433, 23437, 23443, 23445, 23469, 23477, 23488, 23489, 
            23511, 23536, 23538, 23553, 23559, 23568, 23590, 23659, 23667, 23707, 24001, 24151, 24163, 24199, 24211, 24249, 
            24262, 24275, 24289, 24292, 24323, 24416, 24419, 24635, 24653, 24722, 24742, 24789, 25141
        ]
        data['geometry'][invalid_rows] = data['geometry'][invalid_rows].buffer(0)

    
    """ Create dual graphs
    """
    # create dual graph of California
    if make_objects:
        graph_california = Graph.from_geodataframe(data)
        with open('../objects/graphs/voting/california_graph.pkl', 'wb') as f:  pkl.dump(graph_california, f)
    else:
        with open('../objects/graphs/voting/california_graph.pkl', 'rb') as f:  graph_california = pkl.load(f)

    # define NorCal, Cal, SoCal by county fips
    # names: https://sf.curbed.com/2018/6/14/17464134/three-californias-tim-draper-ballot-iniative
    # fips: https://www.weather.gov/hnx/cafips
    # San Benito, Monterey, San Luis Obispo, Santa Barbara, Ventura, Los Angeles
    fips_cal = {'069', '053', '079', '083', '111', '037'}
    # Mono, Madera, Fresno, Kings, Tulare, Inyo, Kern, San Bernardino, Riverside, Orange, San Diego, Imperial
    fips_socal = {'051', '039', '019', '107', '027', '029', '071', '065', '059', '073', '025'}
    # everything else
    if make_objects:
        fips_norcal = set(
            [ 
                fp for fp in data['COUNTYFP']
                if fp not in fips_cal and fp not in fips_socal
            ]
        )
        with open('../objects/fips/voting_norcal_fips.pkl', 'wb') as f:    pkl.dump(fips_norcal, f)
    else:
        with open('../objects/fips/voting_norcal_fips.pkl', 'rb') as f:    fips_norcal = pkl.load(f)


    # create dual graphs of NorCal, Cal, SoCal
    if make_objects:
        graph_norcal, graph_cal, graph_socal = [
            subgraph for subgraph in [
                nx.subgraph(
                    graph_california,
                    [
                        node for node in graph_california.nodes()
                        if graph_california.nodes()[node]['COUNTYFP'] in fips
                    ]
                ) for fips in [fips_norcal, fips_cal, fips_socal]
            ]
        ]
        with open('../objects/graphs/voting/graph_norcal.pkl', 'wb') as f: pkl.dump(graph_norcal, f)
        with open('../objects/graphs/voting/graph_cal.pkl', 'wb') as f: pkl.dump(graph_cal, f)
        with open('../objects/graphs/voting/graph_socal.pkl', 'wb') as f: pkl.dump(graph_socal, f)
    else:
        with open('../objects/graphs/voting/graph_norcal.pkl', 'rb') as f: graph_norcal = pkl.load(f)
        with open('../objects/graphs/voting/graph_cal.pkl', 'rb') as f: graph_cal = pkl.load(f)
        with open('../objects/graphs/voting/graph_socal.pkl', 'rb') as f: graph_socal = pkl.load(f)


    """ Define population variables
    """
    # number of seats for California
    num_districts_california = 52

    # total population
    """ Note: This method bases population on casted votes, such that
              regional population counts rely on the likely false assumption
              that voter turnout is even across regions. As a consequence,
              regional population counts here likely problematically
              underrepresent some regions more than others.
    """
    totpop_california, totpop_norcal, totpop_cal, totpop_socal = [
        sum(
            [
                graph.nodes()[node]['total_votes'] # See note.
                for node in graph.nodes()
            ]
        ) for graph in [graph_california, graph_norcal, graph_cal, graph_socal]
    ]

    # ideal population for California
    idealpop_california = totpop_california / num_districts_california

    # number of seats for NorCal, Cal, SoCal
    num_districts_norcal, num_districts_cal, num_districts_socal = [
        round(num_districts_california * (totpop / totpop_california))
        for totpop in [totpop_norcal, totpop_cal, totpop_socal] 
    ]
    """ Note: Based on computations in make_demographic_ensembles, which uses more reliable Census data,
              the seats for Cal 3 are allocated as NorCal 18, Cal 16, and SoCal 18. The Computations
              above, which use less reliable precinct-level voting data as a proxy for population data,
              find the allocation as NorCal 18, Cal 15, and Socal 18. The sum of these seats is missing 1
              seat from the total of California's 52 seats. As Cal receives 16 seats as opposed to 15 based
              on Census data and that's the only difference, I manually increment the allocation such that Cal
              receives 16 seats and the number of seats across Cal 3 is the same as in California.

              The difference is likely due to rounding.

              As a positive note, the fact that the seat allocations based on voting data and Census data are
              so similar indicates that in this case casted vote counts are a decent proxy for population counts
              on the regional level of NorCal, Cal, and SoCal. However, this does not necessarily guarantee the same
              for the regional level of precincts, which is still important for the representativity of the ensembles.
    """
    # the check below fails without this
    num_districts_cal += 1 
    # make sure no total seats are lost or gained due to rounding
    if sum([num_districts_norcal, num_districts_cal, num_districts_socal]) != num_districts_california:
        raise Exception(f"Total number of seats across Cal 3 doesn't match total number of seats for California.\n\t\
            NorCal: {num_districts_norcal}\n\t\
                Cal: {num_districts_cal}\n\t\
                    SoCal: {num_districts_socal}")

    # ideal population for NorCal, Cal, SoCal
    idealpop_norcal, idealpop_cal, idealpop_socal = [
        totpop / num_districts
        for totpop, num_districts in zip(
            [totpop_norcal, totpop_cal, totpop_socal],
            [num_districts_norcal, num_districts_cal, num_districts_socal]
        )
    ]


    """ Initialize partitions 
    """
    # define updaters
    my_updaters = {
        'cut_edges': cut_edges,
        'district_totvotes': Tally('total_votes', alias='district_totvotes'),
        'district_democrat_votes': Tally('democrat_votes', alias='district_democrat_votes'),
        'district_republican_votes': Tally('republican_votes', alias='district_republican_votes')
    }

    # create three random partitions for California
    part_california_1, part_california_2, part_california_3 = __random_partitions(
        graph_california,
        num_districts_california,
        idealpop_california,
        updaters=my_updaters,
        seeds=[0, 3, 4],
        totpop_key='total_votes'
    )

    # create two random partitions for NorCal, Cal, SoCal each
    part_norcal_1, part_norcal_2 = __random_partitions(
        graph_norcal,
        num_districts_norcal,
        idealpop_norcal,
        updaters=my_updaters,
        seeds=[0, 1],
        totpop_key='total_votes'
    )

    part_cal_1, part_cal_2 = __random_partitions(
        graph_cal,
        num_districts_cal,
        idealpop_cal,
        updaters=my_updaters,
        seeds=[0, 1],
        totpop_key='total_votes'
    )

    part_socal_1, part_socal_2 = __random_partitions(
        graph_socal,
        num_districts_socal,
        idealpop_socal,
        updaters=my_updaters,
        seeds=[0, 1],
        totpop_key='total_votes'
    )


    """ Prepare random walks 
    """
    pop_tolerance = 0.02

    # define proposal methods
    proposal_california, proposal_norcal, proposal_cal, proposal_socal = [
        partial(
            recom,
            pop_col='total_votes',
            pop_target=idealpop,
            epsilon=pop_tolerance,
            node_repeats=1
        ) for idealpop in [idealpop_california, idealpop_norcal, idealpop_cal, idealpop_socal]
    ]

    # define population constraints
    constraint_california, constraint_norcal, constraint_cal, constraint_socal = [
        constraints.within_percent_of_ideal_population(
            partition,
            pop_tolerance,
            pop_key='district_totvotes'
        ) for partition in [part_california_1, part_norcal_1, part_cal_1, part_socal_1]
    ]

    # initialize random walks
    def __markov_chain(proposal, constraint, initial_state, steps=steps):
        return MarkovChain(
            proposal=proposal,
            constraints=[constraint],
            accept=always_accept,
            initial_state=initial_state,
            total_steps=steps
        )

    rw_california_1, rw_california_2, rw_california_3 = [
        __markov_chain(
            proposal_california,
            constraint_california,
            partition,
        ) for partition in [part_california_1, part_california_2, part_california_3]
    ]
    rw_norcal_1, rw_norcal_2 = [
        __markov_chain(
            proposal_norcal,
            constraint_norcal,
            partition
        ) for partition in [part_norcal_1, part_norcal_2]
    ]
    rw_cal_1, rw_cal_2 = [
        __markov_chain(
            proposal_cal,
            constraint_cal,
            partition
        ) for partition in [part_cal_1, part_cal_2]
    ]
    rw_socal_1, rw_socal_2 = [
        __markov_chain(
            proposal_socal,
            constraint_socal,
            partition
        ) for partition in [part_socal_1, part_socal_2]
    ]


    """ Do random walks 
    """
    def __walk(rw, num_districts):
        cutedges_ens = []
        republican_seats_ens = []
        efficiency_gap_ens = []
        for part in rw:
            seats = 0
            
            gap = 0
            wasted_democrat = 0
            wasted_republican = 0
            plan_totvotes = 0
            for i in range(num_districts):
                # count seats won by republicans, and
                # count wasted votes 
                totvotes = part['district_totvotes'][i]
                votes_needed = totvotes / 2
                votes_republican = part['district_republican_votes'][i]
                votes_democrat = part['district_democrat_votes'][i]
                # if republicans win
                if votes_republican / totvotes > 0.5:
                    # count seat
                    seats += 1
                    # count wasted votes
                    wasted_republican += votes_republican - votes_needed
                    wasted_democrat += votes_democrat
                # if democrats win
                elif votes_democrat / totvotes > 0.5:
                    # count wasted votes
                    wasted_republican += votes_republican
                    wasted_democrat += votes_democrat - votes_needed

                plan_totvotes += totvotes

            # compute efficiency gap
            gap = (wasted_democrat - wasted_republican) / plan_totvotes

            cutedges_ens.append(len(part['cut_edges']))
            republican_seats_ens.append(seats)
            efficiency_gap_ens.append(gap)
        return cutedges_ens, republican_seats_ens, efficiency_gap_ens

    print('Walking...', end='\n\t')
    start = timeit.default_timer()

    # ensembles of number of cut edges, of number of republican seats, of efficiency gap
    # California
    cutedges_california_1, republican_seats_california_1, efficiency_gap_california_1 = __walk(rw_california_1, num_districts_california)
    print(f'1/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_california_2, republican_seats_california_2, efficiency_gap_california_2 = __walk(rw_california_2, num_districts_california)
    print(f'2/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_california_3, republican_seats_california_3, efficiency_gap_california_3 = __walk(rw_california_3, num_districts_california)
    print(f'3/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()

    # NorCal
    cutedges_norcal_1, republican_seats_norcal_1, efficiency_gap_norcal_1 = __walk(rw_norcal_1, num_districts_norcal)
    print(f'4/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_norcal_2, republican_seats_norcal_2, efficiency_gap_norcal_2 = __walk(rw_norcal_2, num_districts_norcal)
    print(f'5/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()

    # Cal
    cutedges_cal_1, republican_seats_cal_1, efficiency_gap_cal_1 = __walk(rw_cal_1, num_districts_cal)
    print(f'6/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_cal_2, republican_seats_cal_2, efficiency_gap_cal_2 = __walk(rw_cal_2, num_districts_cal)
    print(f'7/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()

    # SoCal
    cutedges_socal_1, republican_seats_socal_1, efficiency_gap_socal_1 = __walk(rw_socal_1, num_districts_socal)
    print(f'8/9, {round((timeit.default_timer() - start) / 60, 2)} min', end='\n\t')
    start = timeit.default_timer()
    cutedges_socal_2, republican_seats_socal_2, efficiency_gap_socal_2 = __walk(rw_socal_2, num_districts_socal)
    print(f'9/9, {round((timeit.default_timer() - start) / 60, 2)} min')


    """ Aggregate NorCal, Cal, SoCal ensembles into Cal 3 ensembles
    """
    # number of cut edges
    cutedges_cal3_1 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            cutedges_norcal_1, cutedges_cal_1, cutedges_socal_1
        )
    ]
    cutedges_cal3_2 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            cutedges_norcal_2, cutedges_cal_2, cutedges_socal_2
        )
    ]
    # number of republican seats
    republican_seats_cal3_1 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            republican_seats_norcal_1, republican_seats_cal_1, republican_seats_socal_1
        )
    ]
    republican_seats_cal3_2 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            republican_seats_norcal_2, republican_seats_cal_2, republican_seats_socal_2
        )
    ]
    # efficiency gaps
    efficiency_gap_cal3_1 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            efficiency_gap_norcal_1, efficiency_gap_cal_1, efficiency_gap_socal_1
        )
    ]
    efficiency_gap_cal3_2 = [
        norcal + cal + socal for norcal, cal, socal in zip(
            efficiency_gap_norcal_2, efficiency_gap_cal_2, efficiency_gap_socal_2
        )
    ]


    """ Pickle ensembles
    """
    def __pickle(category, object, name):
        with open(f'../ensembles/voting/{category}/{name}.pkl', 'wb') as f:
            pkl.dump(object, f)

    __pickle('cut_edges', cutedges_california_1, 'cutedges_california_1')
    __pickle('cut_edges', cutedges_california_2, 'cutedges_california_2')
    __pickle('cut_edges', cutedges_california_3, 'cutedges_california_3')

    __pickle('cut_edges', cutedges_cal3_1, 'cutedges_cal3_1')
    __pickle('cut_edges', cutedges_cal3_2, 'cutedges_cal3_2')

    __pickle('republican_seats', republican_seats_california_1, 'republican_seats_california_1')
    __pickle('republican_seats', republican_seats_california_2, 'republican_seats_california_2')
    __pickle('republican_seats', republican_seats_california_3, 'republican_seats_california_3')

    __pickle('republican_seats', republican_seats_cal3_1, 'republican_seats_cal3_1')
    __pickle('republican_seats', republican_seats_cal3_2, 'republican_seats_cal3_2')

    __pickle('efficiency_gap', efficiency_gap_california_1, 'efficiency_gap_california_1')
    __pickle('efficiency_gap', efficiency_gap_california_2, 'efficiency_gap_california_2')
    __pickle('efficiency_gap', efficiency_gap_california_3, 'efficiency_gap_california_3')

    __pickle('efficiency_gap', efficiency_gap_cal3_1, 'efficiency_gap_cal3_1')
    __pickle('efficiency_gap', efficiency_gap_cal3_2, 'efficiency_gap_cal3_2')


    """ Print information
    """
    print('Info:', end='\n\t')
    print(f'Number of steps = {steps}', end='\n\t')

    # total votes for Republicans
    total_republican_votes = sum(
        [
            graph_california.nodes()[node]['republican_votes']
            for node in graph_california.nodes()
        ]
    )

    print(f'Percent Republican votes in California: {round((total_republican_votes / totpop_california) * 100, 2)}', end='\n\t')

    print(f'Number of seats:', end='\n\t\t')
    print(f'California = {num_districts_california}', end='\n\t\t')
    print(f'NorCal = {num_districts_norcal}', end='\n\t\t')
    print(f'Cal = {num_districts_cal}', end='\n\t\t')
    print(f'SoCal = {num_districts_socal}')

    print('...Generated voting ensembles.')
    

def main():
    total_start = timeit.default_timer()

    # Use true if objects have not been made yet or are not present in directories.
    # Not having to remake objects saves a couple minutes, which is helpful for development.
    make_objects = False
    n = 50000
    make_demographic_ensembles(n, make_objects=make_objects)
    make_voting_ensembles(n, make_objects=make_objects)

    print(f'Total time: {round((timeit.default_timer() - total_start) / 60 / 60, 2)} hrs')


if __name__ == '__main__':
    main()