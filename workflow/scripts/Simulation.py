"""
Purpose: 
    Simulation 

Author: Munjung Kim
"""


import scipy
import sys
import node2vecs
import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
import os
import argparse
import pandas as pd
import torch.nn as nn


if __name__ == "__main__":
    logging.basicConfig(filename = 'Simulation.log', level = logging.INFO, format='%(asctime)s %(message)s')
    


    strategy = cond_list[trial_index][0] 
    env_dims = cond_list[trial_index][2]
    measurement_capacity_cond = cond_list[trial_index][1]
    if measurement_capacity_cond == "n_dimensions":
        measurement_capacity = env_dims
    elif measurement_capacity_cond == "n_dimensions/2":
        measurement_capacity = int(env_dims//2)
    env_clusters = cond_list[trial_index][3]
    explanation_capacity = cond_list[trial_index][4]
    collective_strategy = cond_list[trial_index][5]
    n_scientists = cond_list[trial_index][6]

    parameters = {"n_scientists": n_scientists, "n_dimensions": env_dims,
                 "dim_length": dim_length, "wishart_scale": wishart_scale, "n_clusters": env_clusters,
                 "ag_max_dimension": env_dims, "measurement_capacity": measurement_capacity,
                 "exp_control_strategy": exp_control_strategy, "experimentation_strategy": strategy,
                 "measurement_strategy": measurement_strategy, "explanation_strategy": explanation_strategy,
                 "explanation_capacity": explanation_capacity, "collective_strategy": collective_strategy}

    # safe_risky_simulation
    print(parameters)

    local_performance = []
    global_performance = []

    # creating a multivariate gaussian environment
    env = clustered_multivariate_gaussian(n_dims=env_dims, max_loc=dim_length, num_clusters=env_clusters, wishart_scale=wishart_scale)

    group = []
    for _ in range(n_scientists):
        scientist_agent = scientist(env_dims, measurement_capacity, exp_control_strategy, strategy, measurement_strategy, 
                             explanation_strategy, explanation_capacity, dim_length)

        scientist_agent.initialize_explanation()
        group.append(scientist_agent)


    # scientists have 300 steps
    for i in range(301):
        print(i)
        agents = np.random.choice(n_scientists,2,replace=False)
        current_scientist = group[agents[0]]
        obs = current_scientist.make_observation(env, group[agents[1]]) # passing another agent in case the strategy is disagreement sampling
        current_scientist.update_explanation()

        if collective_strategy == "full data sharing":
            # shares data with everyone
            for s in range(n_scientists):
                if group[s] == current_scientist:
                    continue
                current_scientist = group[s]
                # the observation that was just collected gets shared with all other agents
                current_scientist.update_data(obs)
                # the new agents update their explanations given the new observation
                current_scientist.update_explanation()


        elif "some data sharing" in collective_strategy:
            # one other scientist learns about this experiment 
            # choose a new scientist: can be anyone except for the one who conducted the experiment
            colleague_idx = random.choice(list(range(0, agents[0])) + list(range(agents[0], n_scientists)))
            colleague = group[colleague_idx]

            colleague.update_data(obs)
            # the new agent updates their explanation given the observation
            colleague.update_explanation()

        # teaching and learning at every step
        if ("teaching and learning" in collective_strategy): 
            agents = np.random.choice(n_scientists,2,replace=False)
            scientist1 = group[agents[0]]
            scientist2 = group[agents[1]]

            if len(scientist1.data) >= 2: 
                teach_learn(scientist1, scientist2)


        # each 10 steps, 2 randomly chosen agents share their explanations
        elif (i>0 and i%10 == 0) and ("explanation sharing" in collective_strategy or "feature sharing" in collective_strategy):
            # choosing two agents who will exchange explanations
            agents = np.random.choice(n_scientists,2,replace=False)
            scientist1 = group[agents[0]]
            scientist2 = group[agents[1]]


            if "aligned explanation sharing" in collective_strategy:
                exchange_explanations_aligned(scientist1, scientist2)

            elif "feature sharing" in collective_strategy:
                exchange_features(scientist1, scientist2)

            elif "aligned skeptical explanation sharing" in collective_strategy:
                exchange_explanations_aligned_skeptical(scientist1, scientist2)



        # record the performance every 5 steps (BUT NOT AT THE VERY FIRST STEP)
        if (i>0 and i%5 == 0):
            local_performances = []
            global_performances = []
            for s in range(n_scientists):
                current_scientist = group[s]
                try:
                    local_performances.append(current_scientist.evaluate_on_collected_data())
                except: 
                    local_performances.append(np.nan)

                global_performances.append(evaluate_performance(current_scientist, env))


            local_performance.append(local_performances)
            global_performance.append(global_performances)

    d = [local_performance, global_performance, [], group, env]

    d.append(parameters)

    with open("epistemology/sim_39/{}.pkl".format(trial_index), "wb") as fp:   #Pickling
        pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)