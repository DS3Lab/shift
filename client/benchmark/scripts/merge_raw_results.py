import os
import json

import pandas as pd

raw_results_fpath = '.cache/raw/vtab-1k/'

results = [x for x in os.listdir(raw_results_fpath) if x.endswith('.json')]

tabular_results = []
for each in results:
    with open(os.path.join(raw_results_fpath, each), 'r') as f:
        tabular_results.append(json.load(f))

tabular_results = pd.DataFrame(tabular_results)
tabular_results.to_csv('./.cache/vtab-1k.csv', index=False)