from typing import Sequence

import pytest
from pydantic.error_wrappers import ValidationError
from schemas._base import ID, generate_id, get_hash
from schemas.models.image_predefined import (
    PredefinedTFImageModelName,
    TFImageModelConfig,
)
from schemas.requests.common import (
    Change,
    ChangeReader,
    ClassifierRequest,
    MutableData,
    MutableReader,
    Request,
)
from schemas.requests.reader import TFReaderConfig

from ..classifier import Classifier


@pytest.fixture
def reader_config():
    r = TFReaderConfig.construct()
    r.embed_feature_path = ["image"]
    r.label_feature_path = ["label"]
    return r


@pytest.fixture
def reader_config_embed_feature_only(reader_config):
    reader_config_ = reader_config.copy(deep=True)
    reader_config_.label_feature_path = None
    return reader_config_


@pytest.fixture
def reader_config_label_feature_only(reader_config):
    reader_config_ = reader_config.copy(deep=True)
    reader_config_.embed_feature_path = None
    return reader_config_


@pytest.fixture
def model_config_1():
    return TFImageModelConfig(tf_image_name=PredefinedTFImageModelName.EFFICIENTNET_B7)


@pytest.fixture
def model_config_2():
    return TFImageModelConfig(tf_image_name=PredefinedTFImageModelName.EFFICIENTNET_B0)


@pytest.fixture
def mutable_reader(reader_config) -> MutableReader:
    return MutableReader(
        reader=reader_config,
        changes=[
            ChangeReader(reader=reader_config, indices=[1, 2, 3]),
            ChangeReader(reader=reader_config, indices=[10, 20, 30]),
        ],
    )


@pytest.fixture()
def mutable_reader_no_changes(reader_config) -> MutableReader:
    return MutableReader(reader=reader_config)


@pytest.fixture
def classifier_1():
    return Classifier.EUCLIDEAN_NN


@pytest.fixture
def classifier_2():
    return Classifier.COSINE_NN


class TestChangeReader:
    def test_validation_indices_positive(self, reader_config):
        # Valid
        _ = ChangeReader(reader=reader_config, indices=[1, 2, 3])

        # Invalid
        with pytest.raises(ValidationError):
            _ = ChangeReader(reader=reader_config, indices=[-2, -1, 0, 1, 2, 3, 4])

    def test_validation_indices_strictly_monotonically_increasing(self, reader_config):
        # Valid
        _ = ChangeReader(reader=reader_config, indices=[1, 2, 3])

        # Invalid
        with pytest.raises(ValidationError):
            _ = ChangeReader(reader=reader_config, indices=[0, 1, 2, 2, 3])
        with pytest.raises(ValidationError):
            _ = ChangeReader(reader=reader_config, indices=[0, 1, 2, 0, 1, 2])


def test_mutable_reader_validation_specified_embed_and_label_feature(
    reader_config, reader_config_embed_feature_only, reader_config_label_feature_only
):
    # 0. Both embed feature and label specified
    _ = MutableReader(reader=reader_config)

    # 1. Only embed feature specified
    with pytest.raises(ValidationError):
        _ = MutableReader(reader=reader_config_embed_feature_only)

    # 2. Only label feature specified
    with pytest.raises(ValidationError):
        _ = MutableReader(reader=reader_config_label_feature_only)


class TestRequest:
    def test_validation_request(self, mutable_reader, model_config_1, classifier_1):
        # 0. Valid
        r = Request(
            train=[mutable_reader],
            test=[mutable_reader],
            models=[model_config_1],
            classifiers=[classifier_1],
        )

        # 1. Cannot have a classifier without test data
        with pytest.raises(ValidationError):
            _ = Request(
                train=[mutable_reader],
                test=None,
                models=[model_config_1],
                classifiers=[classifier_1],
            )

        # 2. Cannot have test data without a classifier
        with pytest.raises(ValidationError):
            _ = Request(
                train=[mutable_reader],
                test=[mutable_reader],
                models=[model_config_1],
                classifiers=None,
            )

        # 3. Cannot alter 'classifiers' into invalid state
        with pytest.raises(ValidationError):
            r.classifiers = None

        # 4. Cannot alter 'test' into invalid state
        with pytest.raises(ValidationError):
            r.test = None

    def test_validation_unique_classifiers(
        self, mutable_reader, model_config_1, classifier_1
    ):
        with pytest.raises(ValidationError):
            _ = Request(
                train=[mutable_reader],
                test=[mutable_reader],
                models=[model_config_1],
                classifiers=[classifier_1, classifier_1],
            )

    def test_generate_request_no_classifier(
        self, mutable_reader, model_config_1, model_config_2
    ):
        request = Request(
            train=[mutable_reader, mutable_reader],
            models=[model_config_1, model_config_2],
        )
        inference_requests, classifier_requests = request.generate_requests([16, 32])
        assert len(classifier_requests) == 0
        # 2 x (base + 2 changes) x 2 models
        assert len(inference_requests) == 12

        # Check the order of models (should be alternating)
        for i in range(len(inference_requests)):
            ir = inference_requests[i]
            if i % 2 == 0:
                assert ir.model == model_config_1
                assert ir.batch_size == 16
            else:
                assert ir.model == model_config_2
                assert ir.batch_size == 32

    def test_generate_request_with_classifier(
        self,
        mutable_reader_no_changes,
        model_config_1,
        model_config_2,
        classifier_1,
        classifier_2,
    ):
        request = Request(
            train=[mutable_reader_no_changes],
            test=[mutable_reader_no_changes],
            models=[model_config_1, model_config_2],
            classifiers=[classifier_1, classifier_2],
        )
        inference_requests, classifier_requests = request.generate_requests([16, 32])
        # 2 models x 2 classifiers
        assert len(classifier_requests) == 4
        # 2 readers (no changes) x 2 models
        assert len(inference_requests) == 4

        # Check order of models
        ir_m_1_test, ir_m_2_test, ir_m_1_train, ir_m_2_train = inference_requests
        assert ir_m_1_train.model == ir_m_1_test.model == model_config_1
        assert ir_m_2_train.model == ir_m_2_test.model == model_config_2

        # Check batch sizes
        assert ir_m_1_train.batch_size == 16 and ir_m_1_test.batch_size == 16
        assert ir_m_2_train.batch_size == 32 and ir_m_2_test.batch_size == 32

        # Check the order of classifiers
        cr_1, cr_2, cr_3, cr_4 = classifier_requests
        # 1. Model 1, classifier 1
        assert (
            cr_1.classifier == classifier_1
            and cr_1.train[0].inference_request_id == ir_m_1_train.id
            and cr_1.test[0].inference_request_id == ir_m_1_test.id
        )
        # 2. Model 2, classifier 1
        assert (
            cr_2.classifier == classifier_1
            and cr_2.train[0].inference_request_id == ir_m_2_train.id
            and cr_2.test[0].inference_request_id == ir_m_2_test.id
        )
        # 3. Model 1, classifier 2
        assert (
            cr_3.classifier == classifier_2
            and cr_3.train[0].inference_request_id == ir_m_1_train.id
            and cr_3.test[0].inference_request_id == ir_m_1_test.id
        )
        # 4. Model 2, classifier 2
        assert (
            cr_4.classifier == classifier_2
            and cr_4.train[0].inference_request_id == ir_m_2_train.id
            and cr_4.test[0].inference_request_id == ir_m_2_test.id
        )


class TestClassifierRequest:
    @pytest.fixture
    def inference_request_ids(self) -> Sequence[ID]:
        return [generate_id() for _ in range(7)]

    @pytest.fixture
    def classifier_request(
        self, classifier_1, inference_request_ids
    ) -> ClassifierRequest:

        return ClassifierRequest(
            classifier=classifier_1,
            train=[
                MutableData(
                    inference_request_id=inference_request_ids[0],
                    inference_request_hash=get_hash("a"),
                    changes=[
                        Change(
                            inference_request_id=inference_request_ids[1],
                            inference_request_hash=get_hash("b"),
                            indices=[0, 1],
                            embed_feature_present=True,
                            label_feature_present=True,
                        ),
                        Change(
                            inference_request_id=inference_request_ids[2],
                            inference_request_hash=get_hash("c"),
                            indices=[2, 3],
                            embed_feature_present=False,
                            label_feature_present=True,
                        ),
                    ],
                ),
                MutableData(
                    inference_request_id=inference_request_ids[3],
                    inference_request_hash=get_hash("d"),
                    changes=[],
                ),
            ],
            test=[
                MutableData(
                    inference_request_id=inference_request_ids[4],
                    inference_request_hash=get_hash("e"),
                    changes=[
                        Change(
                            inference_request_id=inference_request_ids[5],
                            inference_request_hash=get_hash("f"),
                            indices=[0],
                            embed_feature_present=False,
                            label_feature_present=True,
                        ),
                        Change(
                            inference_request_id=inference_request_ids[6],
                            inference_request_hash=get_hash("g"),
                            indices=[1],
                            embed_feature_present=False,
                            label_feature_present=True,
                        ),
                    ],
                )
            ],
            id=generate_id(),
        )

    def test_get_inference_request_ids(self, inference_request_ids, classifier_request):
        assert set(inference_request_ids) == set(
            classifier_request.get_inference_request_ids()
        )

    def test_get_request_without_closing_label_changes(
        self, inference_request_ids, classifier_request
    ):
        wo = classifier_request.get_request_without_closing_label_changes()

        # Train changes
        train_1_changes = wo.train[0].changes
        assert len(train_1_changes) == 1
        assert train_1_changes[0].inference_request_id == inference_request_ids[1]

        assert len(wo.train[1].changes) == 0

        # Test changes
        assert len(wo.test[0].changes) == 0

    def test_get_closing_label_changes(self, inference_request_ids, classifier_request):
        train = classifier_request.get_closing_train_label_changes()
        assert 0 in train and 1 in train
        assert len(train[0]) == 1
        assert train[0][0].inference_request_id == inference_request_ids[2]
        assert len(train[1]) == 0

        test = classifier_request.get_closing_test_label_changes()
        assert 0 in test
        assert len(test[0]) == 2
        assert test[0][0].inference_request_id == inference_request_ids[5]
        assert test[0][1].inference_request_id == inference_request_ids[6]

    def test_hash(self, classifier_request):
        # 1. If changes are removed, hash should change
        assert (
            classifier_request.hash
            != classifier_request.hash_without_closing_label_changes
        )

        # 2. If IDs are changed, hash should stay the same
        classifier_request_copy = classifier_request.copy(deep=True)
        classifier_request_copy.test[0].inference_request_id = generate_id()
        classifier_request_copy.train[0].changes[0].inference_request_id = generate_id()
        assert classifier_request.hash == classifier_request_copy.hash

        # 3. If internal hashes are changed, overall hash should change
        classifier_request_copy.test[0].inference_request_hash = get_hash("newhash")
        assert classifier_request.hash != classifier_request_copy.hash
