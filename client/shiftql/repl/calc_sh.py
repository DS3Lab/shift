import click
import math
import itertools
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

@click.command()
@click.option("--data_size", default=0, help="Dataset Size")
@click.option("--num_models", default=0, help="Number of Models")
def main(data_size,num_models):
    budget, chunk = get_bounds(data_size, num_models)
    print(f"budget: {budget}; chunk: {chunk}")


if __name__ == "__main__":
    main()
