from shift.io.api.client import ShiftAPI
from shift.strategies import EnumerationSearchStrategy
from shift.strategies import MetaLearnedSearchStrategy
from shift.strategies import RandomSearchStrategy
from shift.strategies import ShiftSearchStrategy
from shift.strategies import Task2VecSearchStrategy
from shift.strategies import LeepSearchStrategy
from loguru import logger
def search(config):
    shift_api = ShiftAPI()
    strategies = [
        EnumerationSearchStrategy(),
    ]

    if 'meta' in config:
        strategies.append(MetaLearnedSearchStrategy(
            k_m=config['meta']['k_m']
        ))
    
    if 'random' in config:
        strategies.extend([RandomSearchStrategy(k=random_config['k']) for random_config in config['random']])
    
    if 'shift' in config:
        strategies.extend([ShiftSearchStrategy(shift_config) for shift_config in config['shift']])
        
    if 'task2vec' in config:
        strategies.extend([Task2VecSearchStrategy(task2vec_config) for task2vec_config in config['task2vec']])

    if 'leep' in config:
        strategies.extend([LeepSearchStrategy(leep_config) for leep_config in config['leep']])

    datasets = shift_api.get_ft_all_datasets()
    dataset = config['target']['name']
    if not dataset.startswith("vtab") and dataset not in datasets:
        raise ValueError(f"Dataset '{dataset}' not found in database: {datasets}")
    results = []
    logger.info(strategies)
    for strategy in strategies:
        strategy.set_search_config(config)
        to_ft_models, search_time = strategy.search()
        if len(to_ft_models) > 0:
            sum_ft_time = shift_api.get_total_ft_time(dataset, to_ft_models)
            if len(to_ft_models) > 1:
                best_ft_model, best_ft_acc = shift_api.get_best_model(dataset, to_ft_models)
            else:
                best_ft_model = to_ft_models[0]
            
            if best_ft_model is not None:
                results.append({
                    "strategy": strategy.name,
                    "returned models":len(to_ft_models),
                    "best_ft_acc": best_ft_acc,
                    "sum_ft_time": sum_ft_time/3600,
                    "best_model": best_ft_model,
                })
    return results