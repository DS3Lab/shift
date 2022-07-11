import math
import itertools

readers = {
    "flowers102": 1020,
    "caltech101": 3060,
    "cifar-100": 50000,
    "dmlab": 65550,
}

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

def successive_halving(
    num_models,
    data_size,
    chunk,
    budget,
    eta = 2
):
    S_ks = [num_models]
    n = num_models
    r_ks = []
    sample_index = 0
    for k in range(math.ceil(math.log(n, eta))):
        r_k = math.floor(budget/(S_ks[k] * math.ceil(math.log(n, eta))))
        r_ks.append(r_k)
        R_k = list(itertools.accumulate(r_ks))
        print("k={}, r_k={}".format(k, r_k), end=", ")
        print("Pull {} arms {} times, ".format(S_ks[k], R_k[k]), end="")
        sample_index = sample_index + chunk * R_k[k]
        print("Sample Index: {}".format(sample_index), end=', ')
        S_k = math.ceil(S_ks[k]/eta)
        print("{} arms remaining".format(S_k))
        S_ks.append(S_k)

if __name__=="__main__":
    num_models = 64
    for reader in readers:
        print(readers[reader])
        min_budget, min_chunk = get_bounds(readers[reader], num_models)
        print("[{}] Min Budget: {}, Min Chunk Size: {}".format(reader, min_budget, min_chunk))
        successive_halving(num_models, readers[reader], min_chunk, min_budget)