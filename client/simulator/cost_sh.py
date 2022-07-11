import math
from statistics import mean
import numpy as np
import pandas as pd
import itertools

T_load_data = 700
T_nn = 0.1
T_linear = 5
T_O = 800
T_O_extra = T_O

def get_bounds(data_size, num_models, eta=2):
    """
    budget: is determined by the num_models
    we only need to ensure, at the first step, there is enough budget. i.e., r_k>=1
    """
    # in the first step, S_ks[0] = num_models
    budget = num_models * math.ceil(math.log(num_models, eta))
    """
    chunk size is more complex: we need the total number of pulls
    """
    total_pulls = 0
    S_ks = [num_models]
    r_ks = []
    for k in range(math.ceil(math.log(num_models, eta))):
        r_k = math.floor(budget/(S_ks[k] * math.ceil(math.log(num_models, eta))))
        r_ks.append(r_k)
        R_k = list(itertools.accumulate(r_ks))
        total_pulls += R_k[k]
        S_k = math.ceil(S_ks[k]/eta)
        S_ks.append(S_k)
    chunk = math.ceil(data_size/total_pulls)
    return budget, chunk

def simulate_multiple_nosh(num_models, train_samples, test_samples, classifier, cost_df, num_devices):
    average_load_cost = cost_df['load_cost'][:num_models-1].mean()
    average_inference_cost = cost_df['inference_cost'][:num_models-1].mean()

    T_proxy = T_linear
    if classifier == 'Linear':
        T_proxy = T_linear
    else:
        T_proxy = T_nn
    total_cost = num_models * (
        T_O + T_load_data + 2 *  average_load_cost + average_inference_cost * (train_samples + test_samples)  + T_proxy * test_samples
    )
    total_cost = total_cost / num_devices
    return total_cost / 1000

def simulate_multiple_sh(num_models, train_samples, test_samples, classifier, budget, chunk, cost_df, num_devices):
    average_load_cost = cost_df['load_cost'][:num_models-1].mean()
    average_inference_cost = cost_df['inference_cost'][:num_models-1].mean()
    eta = 2
    T_proxy = T_linear
    if classifier == 'Linear':
        T_proxy = T_linear
    else:
        T_proxy = T_nn
    # first handle inference
    T_test_inference = num_models * (
        average_inference_cost * test_samples + T_load_data + average_load_cost
    )

    T_test_inference = T_test_inference / num_devices
    t_average = 0
    t_max = 0
    t_min = 0

    T_train_total = 0
    # first handle average time 
    S_ks = [num_models]
    r_ks = []
    for k in range(math.ceil(math.log(num_models, eta))):
        # in each run...
        r_k = math.floor(budget/(S_ks[k] * math.ceil(math.log(num_models, eta))))
        r_ks.append(r_k)
        R_k = list(itertools.accumulate(r_ks))
        sample_to_process = chunk * R_k[k]
        t_step = S_ks[k] * (
            T_load_data + average_load_cost + average_inference_cost * sample_to_process
            + T_proxy * test_samples + T_O_extra
        )
        t_step = t_step / min(num_devices, S_ks[k])
        T_train_total += t_step
        S_k = math.ceil(S_ks[k]/eta)
        S_ks.append(S_k)
    t_average = (T_train_total + T_test_inference) /  1000
    # the max time
    T_train_total = 0
    S_ks = [num_models]
    r_ks = []

    infer_sorted_df = cost_df.sort_values(by=['inference_cost'], ascending=False)['inference_cost']
    load_sorted_df = cost_df.sort_values(by=['load_cost'], ascending=False)['load_cost']
    
    for k in range(math.ceil(math.log(num_models, eta))):
        # in each run...
        r_k = math.floor(budget/(S_ks[k] * math.ceil(math.log(num_models, eta))))
        r_ks.append(r_k)
        R_k = list(itertools.accumulate(r_ks))
        sample_to_process = chunk * R_k[k]

        average_inference_cost = infer_sorted_df[:S_ks[k]-1].mean()
        average_load_cost = load_sorted_df[:S_ks[k]-1].mean()

        t_step = S_ks[k] * (
            T_load_data + average_load_cost + average_inference_cost * sample_to_process
            + T_proxy * test_samples
        )
        t_step = t_step / min(num_devices, S_ks[k])
        T_train_total += t_step
        S_k = math.ceil(S_ks[k]/eta)
        S_ks.append(S_k)

    t_max = (T_train_total + T_test_inference) /  1000
    # mean
    T_train_total = 0
    S_ks = [num_models]
    r_ks = []

    infer_sorted_df = cost_df.sort_values(by=['inference_cost'])['inference_cost']
    load_sorted_df = cost_df.sort_values(by=['load_cost'])['load_cost']
    
    for k in range(math.ceil(math.log(num_models, eta))):
        # in each run...
        r_k = math.floor(budget/(S_ks[k] * math.ceil(math.log(num_models, eta))))
        r_ks.append(r_k)
        R_k = list(itertools.accumulate(r_ks))
        sample_to_process = chunk * R_k[k]

        average_inference_cost = infer_sorted_df[:S_ks[k]].mean()
        average_load_cost = load_sorted_df[:S_ks[k]].mean()

        t_step = S_ks[k] * (
            T_load_data + average_load_cost + average_inference_cost * sample_to_process
            + T_proxy * test_samples
        )
        t_step = t_step / min(num_devices, S_ks[k])
        T_train_total += t_step
        S_k = math.ceil(S_ks[k]/eta)
        S_ks.append(S_k)
    t_min = (T_train_total + T_test_inference) /  1000

    return t_average, t_max, t_min

def diff_readers(cost, num_models=100):
    num_devices = 8
    results = []
    for i in range(0, 100000, 100):
        if i > 0:
            budget, chunk = get_bounds(i, num_models, 2)
            cost_nosh = simulate_multiple_nosh(num_models, i, math.ceil(i*0.25), 'Linear', cost, 1)
            cost_nosh_multiple = simulate_multiple_nosh(num_models, i, math.ceil(i*0.25), 'Linear', cost, 8)

            cost_sh_mean, cost_sh_max, cost_sh_min = simulate_multiple_sh(num_models, i, math.ceil(i*0.25), 'Linear', budget, chunk, cost, 1)
            cost_sh_mean_multiple, cost_sh_max_multiple, cost_sh_min_multiple = simulate_multiple_sh(num_models, i, math.ceil(i*0.25), 'Linear', budget, chunk, cost, num_devices)
            
            print("samples: {}, cost_nosh: {}, cost_sh_mean: {}, cost_sh_max: {}, cost_sh_min: {}".format(i, cost_nosh, cost_sh_mean, cost_sh_max, cost_sh_min))
            print("samples: {}, cost_nosh_multiple: {}, cost_sh_mean_multiple: {}, cost_sh_max_multiple: {}, cost_sh_min_multiple: {}".format(i, cost_nosh_multiple, cost_sh_mean_multiple, cost_sh_max_multiple, cost_sh_min_multiple))

    results_df = pd.DataFrame(results)
    return results_df

def diff_models(cost):
    num_devices = 8
    results = []
    samples = 50000
    for i in range(2, 100, 1):
        cost_noshes = []
        cost_noshes_multiple = []
        cost_sh_means = []
        cost_sh_means_multiple = []
        cost_sh_maxs = []
        cost_sh_maxs_multiple = []
        cost_sh_mins = []
        cost_sh_mins_multiple = []
        for j in range(200):
            cost_subset = cost.sample(n=i)
            budget, chunk = get_bounds(samples, i, 2)
            cost_nosh = simulate_multiple_nosh(i, samples, math.ceil(samples*0.25), 'Linear', cost_subset, 1)
            cost_nosh_multiple = simulate_multiple_nosh(i, samples, math.ceil(samples*0.25), 'Linear', cost_subset, 8)

            cost_sh_mean, cost_sh_max, cost_sh_min = simulate_multiple_sh(i, samples, math.ceil(samples*0.25), 'Linear', budget, chunk, cost_subset, 1)
            cost_sh_mean_multiple, cost_sh_max_multiple, cost_sh_min_multiple = simulate_multiple_sh(i, samples, math.ceil(samples*0.25), 'Linear', budget, chunk, cost_subset, num_devices)
            
            cost_noshes.append(cost_nosh)
            cost_noshes_multiple.append(cost_nosh_multiple)
            cost_sh_means.append(cost_sh_mean)
            cost_sh_means_multiple.append(cost_sh_mean_multiple)
            cost_sh_maxs.append(cost_sh_max)
            cost_sh_maxs_multiple.append(cost_sh_max_multiple)
            cost_sh_mins.append(cost_sh_min)
            cost_sh_mins_multiple.append(cost_sh_min_multiple)
            
            print("models: {}, cost_nosh: {}, cost_sh_mean: {}, cost_sh_max: {}, cost_sh_min: {}".format(i, cost_nosh, cost_sh_mean, cost_sh_max, cost_sh_min))
            print("models: {}, cost_nosh_multiple: {}, cost_sh_mean_multiple: {}, cost_sh_max_multiple: {}, cost_sh_min_multiple: {}".format(i, cost_nosh_multiple, cost_sh_mean_multiple, cost_sh_max_multiple, cost_sh_min_multiple))
        
    results_df = pd.DataFrame(results)
    results_df.to_csv('processed/cost_models.csv', index=False)

def resnet_copies(cost):
    num_devices = 8
    results = []
    samples = 50000
    cost = cost[cost['model']=='https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4']
    cost = pd.concat([cost]*100, ignore_index=True)
    print(cost)
    
    for i in range(2, 101, 2):
        budget, chunk = get_bounds(samples, i, 2)

        cost_nosh = simulate_multiple_nosh(i, samples, math.ceil(samples*0.25), 'Linear', cost, 1)
        cost_nosh_multiple = simulate_multiple_nosh(i, samples, math.ceil(samples*0.25), 'Linear', cost, num_devices)

        cost_sh_mean, cost_sh_max, cost_sh_min = simulate_multiple_sh(i, samples, math.ceil(samples*0.25), 'Linear', budget, chunk, cost, 1)
        cost_sh_mean_multiple, cost_sh_max_multiple, cost_sh_min_multiple = simulate_multiple_sh(i, samples, math.ceil(samples*0.25), 'Linear', budget, chunk, cost, num_devices)
        
        print("models: {}, cost_nosh: {}, cost_sh_mean: {}, cost_sh_max: {}, cost_sh_min: {}".format(i, cost_nosh, cost_sh_mean, cost_sh_max, cost_sh_min))
        print("models: {}, cost_nosh_multiple: {}, cost_sh_mean_multiple: {}, cost_sh_max_multiple: {}, cost_sh_min_multiple: {}".format(i, cost_nosh_multiple, cost_sh_mean_multiple, cost_sh_max_multiple, cost_sh_min_multiple))
        
    results_df = pd.DataFrame(results)
    results_df.to_csv('processed/cost_models_resnet.csv', index=False)
    

if __name__=="__main__":
    cost = pd.read_csv('resources/cost_model.csv')
    results_df = diff_readers(cost, num_models = 99)
    results_df.to_csv('processed/cost_99.csv', index=False)
    # diff_models(cost)
    # resnet_copies(cost)