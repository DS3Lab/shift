"""
from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf
from pipeline import DataType
from pipeline.model import PreprocessingSpecs
from pipeline.reader import Reader
from schemas.reader import Column, CSVReaderConfig

from .._config import settings
from ._tensorflow import prepare_dataset


class CSVReader(Reader):
    def __init__(
        self,
        config: CSVReaderConfig,
        specs: PreprocessingSpecs,
        batch_size: Optional[int],
    ):
        if config.other_columns is None:
            selected_columns = [config.embed_column]
        else:
            selected_columns = [config.embed_column] + list(config.other_columns)

        # Use user-specified column names and generate remaining ones
        column_names = self._generate_column_names(config.num_columns, selected_columns)

        # Sort selected columns (indices and types) by their indices
        # This ensures that the 'select_columns' parameter is passed correctly
        column_indices = np.array([x.position for x in selected_columns])
        column_types = np.array([x.type.value for x in selected_columns])
        indices_sorted_order = np.argsort(column_indices)
        sorted_column_indices: list = column_indices[indices_sorted_order].tolist()
        sorted_column_types: list = column_types[indices_sorted_order].tolist()

        # Shuffle params
        shuffle, seed, buffer_size = False, None, None
        if config.shuffle_params is not None:
            shuffle = True
            seed = config.shuffle_params.seed
            buffer_size = config.shuffle_params.buffer_size

        data = tf.data.experimental.make_csv_dataset(
            file_pattern=settings.get_input_path_str(config.csv_path),
            batch_size=1,
            column_names=column_names,
            column_defaults=sorted_column_types,
            select_columns=sorted_column_indices,
            field_delim=config.delimiter,
            use_quote_delim=True,
            header=config.header_present,
            num_epochs=1,
            shuffle=shuffle,
            shuffle_buffer_size=buffer_size,
            shuffle_seed=seed,
            # TODO: not sure whether this relevant only if there are multiple files
            num_parallel_reads=1,
            sloppy=False,
        )

        def extract_values(line: Dict) -> Dict:
            for key in line:
                line[key] = line[key][0]
            return line

        # Determine batch size
        if batch_size is not None:
            batch_size_to_use = batch_size

        # Use largest possible batch size to have all samples in a single batch
        else:
            batch_size_to_use = config.num_records

        self._data = prepare_dataset(
            data=data,
            extraction_fn=extract_values,
            specs=specs,
            batch_size=batch_size_to_use,
            slice_=config.slice,
        )

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        return next(self._data)

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT

    @staticmethod
    def _generate_column_names(
        num_columns: int, defined_columns: Sequence[Column]
    ) -> Sequence[str]:
        all_columns = ["" for _ in range(num_columns)]

        # Set user-specified column names
        for column in defined_columns:
            all_columns[column.position] = column.name

        # Determine which column names are used and which indices already have a
        # column name
        defined_columns_names = set([x.name for x in defined_columns])
        defined_indices = set([x.position for x in defined_columns])

        # Name remaining columns with numbers, but make sure that user did not name
        # columns with those numbers
        value = 0
        for index in range(num_columns):
            if index not in defined_indices:
                while str(value) in defined_columns_names:
                    value += 1
                all_columns[index] = str(value)
                value += 1

        return all_columns
"""
