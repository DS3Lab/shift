# ```Rank by```: Extensions of SHiFT

There are more search strategies appearing in research papers, but it usually takes a lot of time to experiment with new search strategies. To tackle this challenge, SHiFT provides an interface for new search strategies to be integrated into the existing system, enabling these new strategies to be used in the same way as the existing ones.

## Create an Extension

Creating a SHiFT extension is as easy as creating a simple Python class. The following is an example of a SHiFT extension:

```python
from extensions.extension import ShiftExtension

class RandomSearchExtension(ShiftExtension):
    def __init__(self):
        super().__init__("random")

    def __call__(self, models, datasets):
        results = models.copy()
        random.shuffle(results)
        results = [{"json_model": x, "err": -1} for x in results]
        return results
```

A SHiFT extension class has the following requirements:

* It must inherit from `ShiftExtension` class. The base class provides some basic attributes, including: `base_data_path`, `base_src_path` to be used by your extension.
* It must have a `__call__` function that takes two arguments: `models` and `datasets`. The `models` is a list of models, and `datasets` is a list of datasets.
* It must return a list of results, which is a list of dictionaries. Each dictionary has the following keys: `json_model`, `err` (the score of the model).

## Install an Extension

Install an extension is as simple as downloading the source code into the `server/userspace/extensions/src` folder. Once you have downloaded the source code, you will need to restart the userspace controller to load the extension.

The userspace controller will automatically load all existing extensions when it starts.

## Request to Rank by an Extension

```sql
explain json 
    rank * from image_models where source='HuggingFace Transformers' ORDER BY err ASC
        trained on "vtab-caltech101-train" 
        tested on task "vtab-caltech101-val" 
    by "random" wait;
```

It is also possible to request extension with parameters. For example,

```sql
explain json 
    rank * from image_models where source='HuggingFace Transformers' ORDER BY err ASC
        trained on "vtab-caltech101-train" 
        tested on task "vtab-caltech101-val" 
    by "random(seed=42, another_param=1)" wait;
```

Extensions should specify what parameters they need.

## Evaluation of Search Strategies

Performance of different search strategies are measured by its speed and the quality of the results. To facilitate the evaluation, SHiFT provides a simulator as an evaluation tool, together with finetune accuracy achieved on the evaluation datasets.

## Limitations

* Caching is not possible for now. Every time you request an extension, it will be executed again.
* There is no task queue for extensions now.
* Extensions will reuse the same CUDA_VISIBLE_DEVICES environment variable as the other parts.