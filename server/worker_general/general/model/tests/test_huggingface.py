import numpy as np
import torch as pt

from .._huggingface import _get_mean_of_relevant_tokens


def test_get_mean_of_relevant_tokens():
    # batch size = 3, 2 tokens, output dimension = 1
    data = pt.as_tensor(
        [
            # Batch 1
            [[0], [1]],
            # Batch 2
            [[2], [3]],
            # Batch 3
            [[4], [5]],
        ],
        dtype=pt.int64,
    )

    attention_mask = pt.as_tensor(
        [
            # Batch 1
            [1, 1],
            # Batch 2
            [0, 1],
            # Batch 3
            [1, 0],
        ],
        dtype=pt.int64,
    )

    result = _get_mean_of_relevant_tokens(
        batch=data, attention_mask=attention_mask, device=pt.device("cpu")
    )

    assert np.allclose(result.numpy(), np.array([[0.5], [3], [4]]))
