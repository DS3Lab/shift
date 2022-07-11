from pathlib import Path

import numpy as np
import pytest
from schemas.requests.reader import MountedNumPyReaderConfig, ResultsNumPyReaderConfig

from .._numpy_io import NumPyReader, NumPyWriter


@pytest.fixture(scope="function")
def target_path(data_path) -> Path:
    return (
        data_path / "fc556f0841a7aa427f2d513d76364b1cfc556f0841a7aa427f2d513d76364b1c"
    )


def test_empty_dataset_written(target_path):
    writer = NumPyWriter(str(target_path.resolve()))
    writer.finalize()

    # Check that folder exists
    assert target_path.exists() and target_path.is_dir()

    # Check that the reader is complete
    dataset_path = target_path / "dataset.txt"
    assert dataset_path.exists() and dataset_path.is_file()

    # Reading is not successful
    with pytest.raises(ValueError):
        _ = NumPyReader(
            config=MountedNumPyReaderConfig(
                embed_feature="image", numpy_path=target_path.name
            ),
            batch_size=None,
        )


def test_adding_too_much_data(target_path):
    writer = NumPyWriter(str(target_path.resolve()))
    with pytest.raises(ValueError):
        writer.add({"images": np.zeros(shape=(20, 2), dtype=np.uint8)})


def test_reading_from_non_existent_folder():
    with pytest.raises(NotADirectoryError):
        _ = NumPyReader(
            config=MountedNumPyReaderConfig(
                embed_feature="image", numpy_path="fake_data"
            ),
            batch_size=None,
        )


def test_reading_invalid_dataset(target_path):
    writer = NumPyWriter(str(target_path.resolve()))

    # Folder gets created
    for _ in range(5):
        writer.add({"images": np.zeros(shape=(5, 2, 2), dtype=np.uint8)})

    with pytest.raises(FileNotFoundError):
        _ = NumPyReader(
            config=MountedNumPyReaderConfig(
                embed_feature="image", numpy_path=target_path.name
            ),
            batch_size=None,
        )


@pytest.mark.parametrize("finalize_dataset", [True, False])
def test_rewriting_dataset(target_path, finalize_dataset):
    """When the same dataset already exists, either successful or failure it should
    not affect the new dataset."""
    # 1. Write 50 rows of zeros - test both for valid and invalid dataset
    writer1 = NumPyWriter(str(target_path.resolve()))
    for _ in range(10):
        writer1.add({"data": np.zeros(shape=(5, 2, 2), dtype=np.uint8)})

    if finalize_dataset:
        writer1.finalize()

    # 1. Write 25 rows of ones and call finalize - valid dataset
    writer2 = NumPyWriter(str(target_path.resolve()))
    for _ in range(5):
        writer2.add({"data": np.ones(shape=(5, 2, 2), dtype=np.uint8)})
    writer2.finalize()

    data = next(
        NumPyReader(
            config=MountedNumPyReaderConfig(
                embed_feature="data", numpy_path=target_path.name
            ),
            batch_size=None,
        )
    )["data"]

    assert np.allclose(data, np.ones(shape=(25, 2, 2), dtype=np.uint8))


def test_writing_inconsistent_shapes(target_path):
    writer = NumPyWriter(str(target_path.resolve()))
    writer.add({"images": np.zeros(shape=(5, 2, 2), dtype=np.uint8)})

    with pytest.raises(ValueError):
        writer.add({"images": np.zeros(shape=(5, 2, 3), dtype=np.uint8)})


def test_writing_inconsistent_keys(target_path):
    writer = NumPyWriter(str(target_path.resolve()))
    writer.add({"images": np.zeros(shape=(5, 2, 2), dtype=np.uint8)})

    with pytest.raises(KeyError):
        writer.add({"image": np.zeros(shape=(5, 2, 3), dtype=np.uint8)})


def test_read_write_numeric(target_path):
    # 1. Generate data
    np.random.seed(0)
    images = np.random.randint(0, 255, size=(20, 2, 2), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=(20,), dtype=np.int64)

    # 2. Write data
    writer = NumPyWriter(str(target_path.resolve()))
    writer.add({"images": images[0:10, :, :], "labels": labels[0:10]})
    writer.add({"images": images[10:20, :, :], "labels": labels[10:20]})
    writer.finalize()

    # 3. Read data
    reader = NumPyReader(
        config=ResultsNumPyReaderConfig(
            embed_feature="images", label_feature="labels", job_hash=target_path.name
        ),
        batch_size=None,
    )

    # 4. Check it is the same
    result = next(reader)
    assert np.allclose(result["images"], images)
    assert np.allclose(result["labels"], labels)


def test_read_write_strings(target_path):
    # 1. Generate data
    # IMPORTANT: have two arrays of different string lengths to test the behaviour
    # of different string lengths
    first = np.array(["this", "is", "just", "some", "text"])
    second = np.array(
        [
            "some",
            "more",
            "text",
            "to",
            "test",
            "the",
            "batching",
            "functionality",
        ]
    )

    # Concatenate will automatically merge correctly, this should not be passed to
    # the writer
    both = np.concatenate([first, second])

    # 2. Write data
    writer = NumPyWriter(str(target_path.resolve()))
    writer.add({"texts": first})
    writer.add({"texts": second})
    writer.finalize()

    # 3. Read data
    reader = NumPyReader(
        config=ResultsNumPyReaderConfig(
            embed_feature="texts", job_hash=target_path.name
        ),
        batch_size=None,
    )

    # 4. Check if it is the same
    result = next(reader)
    assert np.all(both == result["texts"])


def read_and_write_array(target_path_, data: np.ndarray):
    writer = NumPyWriter(str(target_path_.resolve()))
    writer.add({"data": data})
    writer.finalize()

    return next(
        NumPyReader(
            config=ResultsNumPyReaderConfig(
                embed_feature="data", job_hash=target_path_.name
            ),
            batch_size=None,
        )
    )["data"]


def test_read_write_bytes(target_path):
    input_data = np.array(["hello".encode("UTF-8"), "stríng".encode("UTF-8")])
    output_data = read_and_write_array(target_path, input_data)

    assert np.all(output_data == np.array(["hello", "stríng"]))


def test_read_write_bytes_disguised_as_object(target_path):
    input_data = np.array(
        ["hello".encode("UTF-8"), "stríng".encode("UTF-8")], dtype=np.object_
    )
    output_data = read_and_write_array(target_path, input_data)

    assert np.all(output_data == np.array(["hello", "stríng"]))


def test_read_write_int_disguised_as_object(target_path):
    input_data = np.array([10, 20, 30], dtype=np.object_)
    with pytest.raises(TypeError):
        read_and_write_array(target_path, input_data)


def test_correct_batch_sizes(target_path):
    # 1. Write data
    writer = NumPyWriter(str(target_path.resolve()))

    # Length 9 - for batch size 3 this a corner case because 4th batch will have
    # to start reading from the next file
    for data_length in [9, 8, 7, 10]:
        writer.add({"data": np.zeros(shape=(data_length, 2, 2), dtype=np.int8)})
    writer.finalize()

    # 3. Read data
    reader = NumPyReader(
        config=ResultsNumPyReaderConfig(
            embed_feature="data", job_hash=target_path.name
        ),
        batch_size=3,
    )
    batch_sizes = [b["data"].shape[0] for b in reader]
    assert batch_sizes[-1] <= 3
    for batch_size in batch_sizes[:-1]:
        assert batch_size == 3
