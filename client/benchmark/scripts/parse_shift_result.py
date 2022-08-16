import json
import os
import click
import pandas


@click.command()
@click.option('--path')
def run_parse(path):
    raw_content = [{x: json.load(open(os.path.join(path, x), 'r'))}
                   for x in os.listdir(path) if x.endswith('.json')]
    for each in raw_content:
        results = []
        filename = list(each.keys())[0]
        models = each[filename]['models']['Euclidean NN']
        for model in models:
            results.append({
                'err': model['err'],
                'model_name': model['json_model']['hf_name']
            })
        df = pandas.DataFrame(results)
        df.to_csv(os.path.join(path, filename + '.csv'), index=False)

if __name__ == "__main__":
    run_parse()
