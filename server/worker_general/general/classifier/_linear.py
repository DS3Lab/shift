import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pipeline.io as io
import torch as pt
from pipeline import Device
from schemas import READER_EMBED_FEATURE_NAME as REFN
from schemas import READER_LABEL_FEATURE_NAME as RLFN
from schemas import Hash
from schemas.classifier import GradientDescentSettings
from schemas.requests.common import ClassifierRequest, MutableData
from schemas.response import LinearResult


def _linear_classifier(
    train_X: np.ndarray,
    test_X: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    device: Device,
    settings: GradientDescentSettings,
) -> np.ndarray:
    """
    Perform the linear classification.
    Args:
        train_X (np.ndarray): Training points with shape (num_train_points, num_features)
        test_X (np.ndarray): Testing points with shape (num_test_points, num_features)
        train_y (np.ndarray): training labels with shape (num_train_points,)
        test_y (np.ndarray): testing labels with shape (num_test_points,)

    Returns:
        np.ndarray: predicted labels of the testing points, of the shape (num_test_points, 1)
    """
    input_dimension = train_X.shape[1]
    num_classes = len(set(np.concatenate((test_y, train_y))))
    # since we receive numpy arrays, we can convert them to pytorch tensors to utilitize pytorch.
    train_X = pt.from_numpy(train_X).float()
    test_X = pt.from_numpy(test_X).float()
    train_y = pt.from_numpy(train_y).long()
    test_y = pt.from_numpy(test_y).long()
    logging.info(
        "Creating linear classifier, in_dim: {}, out_dim: {}, y_dim: {}".format(
            train_X.shape, num_classes, train_y.shape
        )
    )
    logging.info("Training linear classifier with the parameters {}:".format(settings))
    model = pt.nn.Linear(input_dimension, num_classes)
    optimizer = pt.optim.SGD(model.parameters(), lr=settings.learning_rate)

    if device == Device.GPU:

        train_X = train_X.cuda()
        test_X = test_X.cuda()
        train_y = train_y.cuda()
        test_y = test_y.cuda()

        model = model.to(pt.device("cuda"))
        try:
            for it in range(settings.num_epochs):
                optimizer.zero_grad()
                outputs = model(train_X)
                loss = pt.nn.CrossEntropyLoss()(outputs, train_y)
                loss.backward()
                optimizer.step()
            outputs_test = model(test_X)
        except RuntimeError as e:
            raise e
    else:
        for it in range(settings.num_epochs):
            optimizer.zero_grad()
            outputs = model(train_X)
            loss = pt.nn.CrossEntropyLoss()(outputs, train_y)
            loss.backward()
            optimizer.step()
        outputs_test = model(test_X)
    return outputs_test.detach().cpu().numpy()


def _get_linear_result(
    train_mutable_data_sequence: Sequence[MutableData],
    test_mutable_data_sequence: Sequence[MutableData],
    device: Device,
    settings: GradientDescentSettings,
) -> LinearResult:

    test_dict, test_md_lengths = io.read_mutable_data_sequence(
        test_mutable_data_sequence
    )

    train_md_dicts = {REFN: [], RLFN: []}
    train_md_length = 0

    # go through the sequence
    # step 1: combine the training points
    # N.B. here we combine all the mutable_data
    for md_index, mutable_data in enumerate(train_mutable_data_sequence):
        train_md_dict, train_md_length = io.read_mutable_data(mutable_data)
        train_md_length += train_md_length
        train_md_dicts[REFN].append(train_md_dict[REFN])
        train_md_dicts[RLFN].append(train_md_dict[RLFN])
    # concate all the features/labels
    train_md_dict[REFN] = np.concatenate(train_md_dicts[REFN], axis=0)
    train_md_dict[RLFN] = np.concatenate(train_md_dicts[RLFN], axis=0)

    test_point_labels = _linear_classifier(
        train_X=train_md_dict[REFN],
        test_X=test_dict[REFN],
        train_y=train_md_dict[RLFN],
        test_y=test_dict[RLFN],
        device=device,
        settings=settings,
    )
    test_point_labels = np.argmax(test_point_labels, axis=1)
    # store results
    return LinearResult(
        test_labels=test_dict[RLFN].tolist(),
        predicted_test_labels=test_point_labels.tolist(),
    )


def linear_classifier(
    request: ClassifierRequest,
    linear_result: Optional[LinearResult],
    device: Device,
    settings: GradientDescentSettings = GradientDescentSettings(),
) -> Dict[Hash, LinearResult]:
    """
    Constructs the linear classifier result that corresponds to the passed classifier request.
    """
    return_value = {}
    if linear_result is None:
        base_request = request
        linear_result = _get_linear_result(
            base_request.train, base_request.test, device=device, settings=settings
        )
        # now determine, for each test point, the predicted label
        return_value[base_request.hash] = linear_result
    return return_value
