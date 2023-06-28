#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: marinadubova

@Modified by:   Munjung Kim
@Last Modified time: 2023-06-27 22:43:30

"""

import numpy as np
import keras
from keras import layers


def sdimlists(dimsvals, return_full=True):
    assert isinstance(dimsvals, list), "Dims and values are not a list!"

    return {"cond_dims": None, "cond_vals": None} if not dimsvals else {"cond_dims": np.array(dimsvals[0]),
                                                                        "cond_vals": np.array(dimsvals[1]),
                                                                        "return_full": return_full}

def evaluate_performance(scientist, environment, n=1000):
    score = None
    # how many observations to sample
    ground_truth_sample = []
    for i in range(n):
        obs = environment.sample()
        ground_truth_sample.append(obs)
    ground_truth_sample = np.array(ground_truth_sample)
    if scientist.explanation_strategy == "nn_autoencoder":
        score = scientist.explanation.evaluate(ground_truth_sample, ground_truth_sample)
        
    return score


class scientist:
    def __init__(self, max_dimensions, measurement_capacity, exp_control_strategy,
                experimentation_strategy, measurement_strategy, explanation_strategy,
                explanation_capacity, max_dim_value = float('inf')):
        """
        Initializing clustered multivariate gaussian ground truth

        Args:
            max_dimensions :
            experimentation_strategy : random, safe, safe_probabilistic, risky, risky_probabilistic, disagreement_sampling, disagreement_sampling_probabilistic, and disagreement_safe_probabilistic
        """
        self.explanation = None
        self.data = []
        self.max_dimensions = max_dimensions # dimensionality of the world!
        self.max_dim_value = max_dim_value
        self.minimum_exploration = 0.1
        
        self.dimension_importance_weights = np.ones((self.max_dimensions)) # not normalized
        # how important are the dimensions to measure
        # here, the agent is agnostic in the beginning: all the dimensions are equally important
        
        self.measurement_capacity = measurement_capacity # max number of dimensions to record at each experiment
        
        self.exp_control_strategy = exp_control_strategy
        self.experimentation_strategy = experimentation_strategy 
        
        
        
    def make_observation(self, env, scientist2=None):
        
        data = np.array(self.data)
        
        # what to measure if there is no data yet
        if len(data) < 10 or np.random.random() < self.minimum_exploration or self.experimentation_strategy == "random": # exploration here!
            data_indices = self.pick_dimensions_random("maximum_random") # is this any different from experimentation strategy???
            #controlled_dim = [np.random.choice(data_indices)]
            #exp_value = [np.random.uniform(env.dim_length)]
            #experiment_parameters = [controlled_dim, exp_value]
            experiment_parameters = []
        
        # exp_control_strategy: which dimensions are fixed in the experiment
        # only influences if agents actually want to control something
        # TODO
        else:
        
            if self.experimentation_strategy == "safe":
                if self.explanation_strategy == "nn_autoencoder":
                    res = []
                    for i in range(len(data)):
                        res.append(self.explanation.evaluate(data[i:i+1], data[i:i+1]))
                    target_observation_idx = np.argmin(res)
                    best_explained_dims = np.where(data[target_observation_idx]!=0.)[0] # the most well predicted observation
                    data_indices = best_explained_dims # record the same values as in the best observation
                    
            elif self.experimentation_strategy == "safe_probabilistic": 
                if self.explanation_strategy == "nn_autoencoder":
                    res = []
                    for i in range(len(data)):
                        res.append(- self.explanation.evaluate(data[i:i+1], data[i:i+1])) # minus the error - because we want to sample observations more often if they have a smaller prediction error
                    target_observation_idx = random.choices(range(len(data)), cum_weights = res)[0] # the observations are chosen accordingly to their prediction error
                    best_explained_dims = np.where(data[target_observation_idx]!=0.)[0] 
                    data_indices = best_explained_dims # record the same values as in the best observation
                    

            # choosing values for the fixed dimensions
            # only influences if agents actually want to control something
            # TODO
            
            elif self.experimentation_strategy == "risky":
                if self.explanation_strategy == "nn_autoencoder":
                    res = []
                    for i in range(len(data)):
                        res.append(self.explanation.evaluate(data[i:i+1], data[i:i+1]))
                    target_observation_idx = np.argmax(res)
                    best_explained_dims = np.where(data[target_observation_idx]!=0.)[0] # the observation with the most loss
                    data_indices = best_explained_dims # record the same values as in the best observation
            
            elif self.experimentation_strategy == "risky_probabilistic":
                if self.explanation_strategy == "nn_autoencoder":
                    res = []
                    for i in range(len(data)):
                        res.append(self.explanation.evaluate(data[i:i+1], data[i:i+1]))
                    target_observation_idx = random.choices(range(len(data)), cum_weights = res)[0] # the observations are sampled proportionally to their prediction error
                    best_explained_dims = np.where(data[target_observation_idx]!=0.)[0] # the observation with the most loss
                    data_indices = best_explained_dims # record the same values as in the best observation
                 
            
           
            elif self.experimentation_strategy == "disagreement_sampling":
                if self.explanation_strategy == "nn_autoencoder":
                    if len(scientist2.data)>0:
                        data = np.concatenate((self.data[:], scientist2.data[:])) # testing disagreement on the full dataset containing both agents' data
                    distances = np.diagonal(cdist(scientist2.explanation.predict(data),self.explanation.predict(data), metric = 'euclidean'))
                    target_observation_idx = np.argmax(distances) # observation for which both agents give the most different predictions
                    target_obs_dims = np.where(data[target_observation_idx]!=0.)[0] # the observation with the most loss
                    data_indices = target_obs_dims # record the same values as in the best observation
            
            elif self.experimentation_strategy == "disagreement_sampling_probabilistic":
                if self.explanation_strategy == "nn_autoencoder":
                    if len(scientist2.data)>0:
                        data = np.concatenate((self.data[:], scientist2.data[:])) # testing disagreement on the full dataset containing both agents' data
                    distances = np.diagonal(cdist(scientist2.explanation.predict(data),self.explanation.predict(data), metric = 'euclidean'))
                    target_observation_idx = random.choices(range(len(data)), cum_weights = distances)[0] # observation for which both agents give the most different predictions
                    target_obs_dims = np.where(data[target_observation_idx]!=0.)[0] # the observation with the most loss
                    data_indices = target_obs_dims # record the same values as in the best observation
            
            elif self.experimentation_strategy == "disagreement_safe_probabilistic":
                if self.explanation_strategy == "nn_autoencoder":
                    res = []
                    if len(scientist2.data)>0:
                        data = np.concatenate((self.data[:], scientist2.data[:])) # testing disagreement on the full dataset containing both agents' data
                    # sorting the datapoints by disagreement
                    distances = np.diagonal(cdist(scientist2.explanation.predict(data),self.explanation.predict(data), metric = 'euclidean'))
                    disagreement_sorted = distances.argsort()
                    disagreement_rankings = np.empty_like(disagreement_sorted)
                    disagreement_rankings[disagreement_sorted] = np.arange(len(distances))
                    # sorting the datapoitns by how well they are accounted for
                    for i in range(len(data)):
                        res.append(- self.explanation.evaluate(data[i:i+1], data[i:i+1])) # minus the error - because we want to sample observations more often if they have a smaller prediction error
                    prediction_error_sorted = np.array(res).argsort()
                    prediction_error_rankings = np.empty_like(prediction_error_sorted)
                    prediction_error_rankings[prediction_error_sorted] = np.arange(len(res))
                    
                    # the observation will be sampled more often if it is well accounted by the scientist's explanation and if it is in disagreement with some other agent
                    combined_rankings = prediction_error_rankings + disagreement_rankings
                    
                    target_observation_idx = random.choices(range(len(data)), cum_weights = combined_rankings)[0] # observation for which both agents give the most different predictions
                    target_obs_dims = np.where(data[target_observation_idx]!=0.)[0] # the observation with the most loss
                    data_indices = target_obs_dims # record the same values as in the best observation
            
            elif self.experimentation_strategy == "disagreement_risky_probabilistic":
                if self.explanation_strategy == "nn_autoencoder":
                    res = []
                    if len(scientist2.data)>0:
                        data = np.concatenate((self.data[:], scientist2.data[:])) # testing disagreement on the full dataset containing both agents' data
                    # sorting the datapoints by disagreement
                    distances = np.diagonal(cdist(scientist2.explanation.predict(data),self.explanation.predict(data), metric = 'euclidean'))
                    disagreement_sorted = distances.argsort()
                    disagreement_rankings = np.empty_like(disagreement_sorted)
                    disagreement_rankings[disagreement_sorted] = np.arange(len(distances))
                    # sorting the datapoitns by how well they are accounted for
                    for i in range(len(data)):
                        res.append(self.explanation.evaluate(data[i:i+1], data[i:i+1])) # sample observations more often if they have bigger prediction error
                    prediction_error_sorted = np.array(res).argsort()
                    prediction_error_rankings = np.empty_like(prediction_error_sorted)
                    prediction_error_rankings[prediction_error_sorted] = np.arange(len(res))
                    
                    # the observation will be sampled more often if it is not well accounted by the scientist's explanation and if it is in disagreement with some other agent
                    combined_rankings = prediction_error_rankings + disagreement_rankings
                    
                    target_observation_idx = random.choices(range(len(data)), cum_weights = combined_rankings)[0] # observation for which both agents give the most different predictions
                    target_obs_dims = np.where(data[target_observation_idx]!=0.)[0] # the observation with the most loss
                    data_indices = target_obs_dims # record the same values as in the best observation
                    
            
            elif self.experimentation_strategy == "sampling_the_unknown":
                if self.explanation_strategy == "nn_autoencoder":
                    # generating the indices for dimensions to control in the potential experiments - generating 30 options
                    dimensions = np.random.uniform(0,self.max_dimensions-1,500).astype(int)
                    # generating values along these dimensions
                    values = np.random.uniform(0,self.max_dim_value,500)
                    data_array = np.array(data) #inefficient! data can be np array from the beginning - maybe change later
                    data_array_filtered_by_dims = data_array[:,dimensions]
                    # for conditions when agents do not record some observations
                    missed_dimensions = np.where(data_array_filtered_by_dims==0.)
                    data_array_filtered_by_dims[missed_dimensions] = np.nan
                    data_diff_from_proposals = np.absolute(data_array_filtered_by_dims - values)
                    distances = np.nanmin(data_diff_from_proposals, axis = 0)
                    max_min_distance_idx = np.argmax(distances)
                    dim_to_control = dimensions[max_min_distance_idx]
                    control_value = values[max_min_distance_idx]
                    if self.measurement_capacity == self.max_dimensions:
                        data_indices = np.arange(self.max_dimensions)
                    # the dimensions to record are also sampled randomly - might be an potential confound
                    else: 
                        data_indices = np.random.choice(self.max_dimensions, self.measurement_capacity, replace = False)
                        if dim_to_control not in data_indices:
                            swap_idx = np.random.choice(self.measurement_capacity,1)
                            data_indices[swap_idx] = dim_to_control
                    
            
            # choosing the control value close to one of the values in target observation
            if self.experimentation_strategy != "none" and self.experimentation_strategy !=  "sampling_the_unknown" and self.exp_control_strategy == "close":                
                dim_to_control = np.random.choice(np.where(data[target_observation_idx]!=0.)[0])
                control_value = min(data[target_observation_idx][dim_to_control] + np.random.uniform(-1,1), self.max_dim_value)
            
            
            experiment_parameters = [[dim_to_control],[control_value]]
            
            # switch uncontrolled dimensions if exploring
            if self.measurement_capacity < self.max_dimensions:
                data_indices = np.array(data_indices)
                switch_boolean = np.logical_and(np.random.random(len(data_indices))<0.1, data_indices!=dim_to_control)
            
                # CHECK THIS
                data_indices[switch_boolean] = np.random.randint(self.max_dimensions, size = len(data_indices))[switch_boolean]
          
        raw_observation = np.array(env.sample(**convert_dimlists(experiment_parameters))) # one observation
        # recording only the dimensions that were measured
        current_observation = np.zeros((len(raw_observation))) # TODO: non-measured dimensions are currently zeros, 
        # check if it can screw up the models
        current_observation[data_indices] = raw_observation[data_indices] #projection
        
        # saving the result
        self.data.append(current_observation)
        
        return current_observation
        
    def pick_dimensions_random(self, strategy):
        dimensions_for_experiment = []
        if strategy == "all":
            dimensions_for_experiment = list(range(self.max_dimensions))
            
        # currently just picking the dimensions randomly at the maximum capacity
        if strategy == "maximum_random":
            dimensions_for_experiment = np.random.choice(range(self.max_dimensions), 
                                                    size = self.measurement_capacity, 
                                                    p = self.dimension_importance_weights/np.sum(self.dimension_importance_weights), # normalizing here
                                                    replace = False)
            
        return dimensions_for_experiment
        
        
    
        
        
class scientist_autoencoder(scientist):
        
    def __init__(self, max_dimensions, measurement_capacity, exp_control_strategy,
                experimentation_strategy, measurement_strategy, explanation_strategy,
                explanation_capacity, num_param, max_dim_value = float('inf')):
        """
        Initializing clustered multivariate gaussian ground truth

        Args:
            max_dimensions :
            experimentation_strategy : random, safe, safe_probabilistic, risky, risky_probabilistic, disagreement_sampling, disagreement_sampling_probabilistic, and disagreement_safe_probabilistic
        """
        self.explanation = None
        self.data = []
        self.max_dimensions = max_dimensions # dimensionality of the world!
        self.max_dim_value = max_dim_value
        self.minimum_exploration = 0.1
        self.explanation_strategy = "nn_autoencoder"
        self.explanation_capacity = explanation_capacity
        
        self.dimension_importance_weights = np.ones((self.max_dimensions)) # not normalized
        # how important are the dimensions to measure
        # here, the agent is agnostic in the beginning: all the dimensions are equally important
        
        self.measurement_capacity = measurement_capacity # max number of dimensions to record at each experiment
        
        self.exp_control_strategy = exp_control_strategy
        self.experimentation_strategy = experimentation_strategy 
        self.num_param : num_param
    
    
    
    def initialize_explanation(self):
        if self.explanation_strategy == "nn_autoencoder":
            encoding_dim = self.explanation_capacity

            input_data = keras.Input(shape=(self.max_dimensions,))
            # "encoded" is the encoded representation of the input
            print(encoding_dim)
            encoded = layers.Dense(encoding_dim, activation='linear')(input_data)
            # "decoded" is the lossy reconstruction of the input
            decoded = layers.Dense(self.max_dimensions, activation='relu')(encoded)

            # This model maps an input to its reconstruction
            autoencoder = keras.Model(input_data, decoded)
        
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            
            self.explanation = autoencoder
        
    def update_explanation(self):
        data = np.array(self.data) # currently explanations are based on all the observations
        
        
        if self.explanation_strategy == "nn_autoencoder":
            
            
            self.explanation.fit(data, data,
                epochs=50,
                batch_size=1,
                shuffle=True,
                validation_data=(data, data), verbose = False)
            
    def evaluate_on_collected_data(self):
        score = self.explanation.evaluate(np.array(self.data), np.array(self.data))
        return score
    
    def update_data(self, datapoint):
        self.data.append(datapoint)
        
        
        


        

    
