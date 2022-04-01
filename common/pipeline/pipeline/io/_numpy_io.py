import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from celery.utils.log import get_task_logger
from pipeline import DataType
from pipeline.reader import Reader
from schemas.requests.reader import NumPyLocation, NumPyReaderConfig

from .._config import settings

_logger = get_task_logger(__name__)


class _TooMuchData(BaseException):
    """Exception used internally to signal that there is too much data to store."""

    pass


class _AllocatedArrays:
    """Manages NumPy data in memory and makes sure that only the specified amount of
    memory is used.

    Args:
        max_rows (int): Maximal number of rows that can be in memory at the same time.
        max_values (int): Maximal number of values (any type) that can be in memory at
            the same time.
        residual_shapes_and_dtypes (Dict[str, Tuple[Tuple[int, ...], np.dtype]]):
            Mapping from key to the shape and type of data that will be stored in
            memory. First dimension is excluded and is used for stacking the data
            points together.
    """

    def __init__(
        self,
        max_rows: int,
        max_values: int,
        residual_shapes_and_dtypes: Dict[str, Tuple[Tuple[int, ...], np.dtype]],
    ):
        # 1. Determine maximal number of rows
        # Maximal number of rows is the minimum of
        # 1. Specified maximal number of rows
        # 2. Maximal number of rows that together do not have more values than the
        # specified maximal number of values
        num_values_per_row = np.sum(
            [np.prod(i[0]) for i in residual_shapes_and_dtypes.values()]
        )
        max_rows_not_exceeding_max_values = int(max_values // num_values_per_row)

        self._occupancy = 0
        self._max_occupancy = min(max_rows, max_rows_not_exceeding_max_values)
        _logger.debug(
            "Maximal number of rows stored in memory while saving results to disk is "
            "%d based on the maximal number of rows %d and maximal number of values %d",
            self._max_occupancy,
            max_rows,
            max_values,
        )

        # 2. Preallocate space in memory
        self._data = {}
        for key in residual_shapes_and_dtypes:
            shape = (self._max_occupancy, *residual_shapes_and_dtypes[key][0])
            dtype = residual_shapes_and_dtypes[key][1]

            # For string and byte types use np.object_
            # so that strings are not limited by length
            # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.str_
            # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bytes_
            if np.issubdtype(dtype, np.unicode_) or np.issubdtype(dtype, np.bytes_):
                dtype = np.object_

            # Since parts of data will be inserted into the array, they will inherit
            # the object dtype
            self._data[key] = np.zeros(shape=shape, dtype=dtype)

    def add(self, dictionary: Dict[str, np.ndarray]):
        """

        Args:
            dictionary (Dict[str, np.ndarray]): Data to be added.
        """
        # 1. Check that keys match
        expected_keys = set(self._data.keys())
        received_keys = set(dictionary.keys())

        if expected_keys != received_keys:
            raise KeyError(
                f"Found different dictionary keys, expected {expected_keys}, "
                f"received {received_keys}"
            )

        # 2. Check that there is enough space to add the new data
        first_key = list(dictionary.keys())[0]
        to_add = dictionary[first_key].shape[0]

        new_size = self._occupancy + to_add

        if to_add > self._max_occupancy:
            raise ValueError(
                f"Tried to store too much data at the same time ({to_add} rows), "
                f"however maximal number of rows is {self._max_occupancy}"
            )

        if new_size > self._max_occupancy:
            _logger.debug(
                "New number of rows %d exceeds the maximal number of rows %d",
                new_size,
                self._occupancy,
            )
            raise _TooMuchData

        # 3. Add the data
        for key in dictionary:
            # Works if array has one or more than one dimension
            try:
                self._data[key][
                    self._occupancy : self._occupancy + to_add
                ] = dictionary[key]
            except ValueError:
                _logger.exception("Shapes of data not the same")
                raise

        self._occupancy += to_add

    def retrieve_data(self) -> Dict[str, np.ndarray]:
        """Retrieve existing data from memory.

        Returns:
            Dict[str, np.ndarray]: Data stored in memory.
        """
        result = {}
        for key in self._data:
            data = self._data[key][: self._occupancy]

            # Convert np.object_ type to string
            if data.dtype == np.object_:
                # TODO: Better way to obtain a sample
                sample = data.item(0)
                if isinstance(sample, str):
                    result[key] = data.astype(np.unicode_)
                elif isinstance(sample, bytes):
                    result[key] = np.char.decode(
                        data.astype(np.bytes_), encoding="UTF-8"
                    )
                else:
                    raise TypeError(f"Cannot process type {type(sample)}")
            else:
                result[key] = data

        self._occupancy = 0
        return result

    @property
    def is_empty(self):
        """Specifies whether not data is stored at the moment."""
        return self._occupancy == 0


class NumPyWriter:
    """Writes the supplied data to multiple NumPy '.npz' files, which  are portable
    across machines.
    For more info see:
    https://numpy.org/devdocs/reference/generated/numpy.lib.format.html

    IMPORTANT: object type will be automatically converted to a string type, so that
    pickling can be disabled.

    Args:
        location (str): Path to the folder where the data should be stored. The
            specified folder should not already exist.
    """

    def __init__(self, location: str):
        self._location = location
        self._allocated_arrays: Optional[_AllocatedArrays] = None
        self._file_counter = 1

    def _prepare_directory(self):
        """Creates directory if it does not exist and removes all files in it."""
        directory_location = Path(self._location)
        directory_location.mkdir(parents=True, exist_ok=True)
        _logger.debug("Created folder %r", self._location)
        for child in directory_location.iterdir():
            _logger.debug("Removing file %s", str(child.resolve()))
            child.unlink()

    def add(self, dictionary: Dict[str, np.ndarray]):
        """Adds data that should be stored to files.

        Args:
            dictionary (Dict[str, np.ndarray]): New data.
        """
        # This is the first time that add has been called
        if self._allocated_arrays is None:
            # 1. Create a folder where data will be stored
            self._prepare_directory()

            # Instantiate the memory management object
            shapes_and_dtypes = {
                key: (dictionary[key].shape[1:], dictionary[key].dtype)
                for key in dictionary
            }
            self._allocated_arrays = _AllocatedArrays(
                max_rows=settings.result_max_rows,
                max_values=settings.result_max_values,
                residual_shapes_and_dtypes=shapes_and_dtypes,
            )

        try:
            self._allocated_arrays.add(dictionary)
        except _TooMuchData:
            self._flush_data()
            self._allocated_arrays.add(dictionary)

    def _flush_data(self):
        """Flushes data from memory to disk (a new file)."""
        result = self._allocated_arrays.retrieve_data()

        # Count is padded with 10 0s
        current_filename = f"{self._file_counter:010}"
        np.savez(
            os.path.join(self._location, current_filename),
            **{str(key): result[key] for key in result},
        )
        _logger.debug("File %r flushed to disk", current_filename)
        self._file_counter += 1

    def finalize(self):
        """Finalizes the dataset, so that it can be used by readers."""

        # 1. Flush remaining if any remaining data exists
        if self._allocated_arrays is not None and not self._allocated_arrays.is_empty:
            # 1. Flush remaining data
            # self._allocated_array does not get created for an empty reader
            self._flush_data()

        # 2. If no data has been supplied, create the folder now
        if self._allocated_arrays is None:
            self._prepare_directory()

        # 3. Write a file that denotes that the dataset is valid
        dataset_file_obj = Path(os.path.join(self._location, "dataset.txt"))
        dataset_file_obj.touch()

        _logger.debug("Folder %r finalized", self._location)


class NumPyReader(Reader):
    """Loads the data stored in multiple '.npz'
    (https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) files.

    Args:
        config (NumPyReaderConfig): Reader configuration.
        batch_size (int, optional): Batch size; if not specified, maximal possible batch
            size (whole dataset) is used.
    """

    def __init__(self, config: NumPyReaderConfig, batch_size: Optional[int]):
        self._batch_size = batch_size

        # 1. Determine files to read
        # # Full path to the folder
        base_location, relative_path = config.data_location
        data_path = (
            settings.get_results_path_str(relative_path)
            if base_location == NumPyLocation.RESULTS
            else settings.get_input_path_str(relative_path)
        )

        # # Ensure folder exists
        data_path_obj = Path(data_path)
        if not data_path_obj.exists() or not data_path_obj.is_dir():
            raise NotADirectoryError(
                f"NumPy dataset with base location {base_location!r} and relative path "
                f"to the dataset {relative_path!r} does not exist!"
            )

        # # Ensure the data is valid
        dataset_file_path_obj = Path(os.path.join(data_path, "dataset.txt"))
        if not dataset_file_path_obj.exists() or not dataset_file_path_obj.is_file():
            raise FileNotFoundError(
                f"File 'dataset.txt' does not exist in a NumPy dataset with base "
                f"location {base_location!r} and relative path to the "
                f"dataset {relative_path!r}. This means that the dataset is not valid "
                f"or not complete."
            )

        # # Find '.npz' files in the folder
        with os.scandir(data_path) as entries_iterator:
            self._filename_paths = sorted(
                [
                    os.path.join(data_path, e.name)
                    for e in entries_iterator
                    if e.is_file() and e.name.endswith(".npz")
                ]
            )

        # # There must be at least one file
        if len(self._filename_paths) == 0:
            raise ValueError(
                "Failed to read {}: There must be at least one file to read from!".format(
                    data_path
                )
            )

        # 2. Keys to use
        self._keys = []
        if config.embed_feature is not None:
            self._keys.append(config.embed_feature)
        if config.label_feature is not None:
            self._keys.append(config.label_feature)

        # 3. Prepare file metadata variables
        # # Which file is currently processed
        self._file_index = 0
        # # What is the position within current file
        self._position_index = 0
        # # What is the size of the current file
        self._file_length = 0
        # # What is the content of the current file
        self._data: Dict[str, np.ndarray] = {}

        # 4. Was any file ever loaded
        self._first_load_completed = False

    def _extract_slice(
        self, start_index: int, end_index: Optional[int]
    ) -> Dict[str, np.ndarray]:
        result = {}
        for key in self._data:
            if end_index is None:
                result[key] = self._data[key][start_index:].copy()
            else:
                result[key] = self._data[key][start_index:end_index].copy()

        return result

    def _load_next_file(self):
        # 1. Load metadata
        filename_to_load = self._filename_paths[self._file_index]
        with np.load(filename_to_load) as file:

            # 2. Load the file data to the memory (load only data that is requested)
            self._data = {}
            file_length: Optional[int] = None
            for key in self._keys:
                if key not in file:
                    raise KeyError(f"Key {key!r} not present in {filename_to_load!r}")
                key_data = file[key]
                if file_length is None:
                    file_length = key_data.shape[0]
                elif file_length != key_data.shape[0]:
                    raise ValueError(
                        f"Data in {filename_to_load!r} does not have same length "
                        f"({file_length} != {key_data.shape[0]})"
                    )
                self._data[key] = key_data
            self._file_length = file_length

        # 3. Update indices
        self._file_index += 1
        self._position_index = 0

    @property
    def _files_exhausted(self) -> bool:
        return self._file_index >= len(self._filename_paths)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        # 1. Check special cases
        # # First iteration - no data has been loaded yet
        if not self._first_load_completed:
            self._load_next_file()
            self._first_load_completed = True

        # # Check if possible to return any more data
        if self._file_length - self._position_index == 0 and self._files_exhausted:
            raise StopIteration

        # 2. Prepare variables
        # # Parts from different files that will be stacked together
        parts: List[Dict[str, np.ndarray]] = []

        # # Number of samples needed for the current batch
        remaining_samples_needed = (
            self._batch_size if self._batch_size is not None else np.inf
        )

        # 3. Iterate through files (one iteration is one file)
        while remaining_samples_needed > 0:
            available_in_current_file = self._file_length - self._position_index

            # # Current file contains enough data - this will be the last processed file
            if remaining_samples_needed <= available_in_current_file:
                # Extract a slice
                parts.append(
                    self._extract_slice(
                        self._position_index,
                        self._position_index + remaining_samples_needed,
                    )
                )

                # Update position
                self._position_index += remaining_samples_needed
                break

            # # Current file does not contain enough data
            else:
                # # Current file contains some data
                if available_in_current_file > 0:
                    parts.append(self._extract_slice(self._position_index, None))
                    remaining_samples_needed -= available_in_current_file
                    self._position_index = self._file_length

                # # Try to load next file
                if self._files_exhausted:
                    break

                self._load_next_file()

        # 4. Concatenate data from multiple files
        return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}

    @property
    def data_type(self) -> DataType:
        """Returns the default type, since type inference is not performed.

        Returns:
            DataType: The default type (type not inferred).
        """
        return DataType.UNKNOWN
