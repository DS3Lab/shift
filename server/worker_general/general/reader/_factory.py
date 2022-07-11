from typing import Optional

from pipeline.model import PreprocessingSpecs
from pipeline.reader import Reader, ReaderFactory
from schemas.requests.reader import (  # CSVReaderConfig,
    HFReaderConfig,
    ImageFolderReaderConfig,
    PTReaderConfig,
    ReaderConfig,
    TFReaderConfig,
    VTABReaderConfig,
)

# from ._csv import CSVReader
from ._huggingface import HFReader
from ._image_folder import ImageFolderReader
from ._tensorflow import TFReader
from ._torchvision import _get_torchvision_reader
from ._vtab import VTABReader


class AllReaderFactory(ReaderFactory):
    @staticmethod
    def get_reader(
        reader_config: ReaderConfig,
        batch_size: Optional[int],
        specs: PreprocessingSpecs,
    ) -> Reader:
        # Order important: HFReaderConfig and ImageFolderReaderConfig subclass
        # PTReaderConfig
        if isinstance(reader_config, HFReaderConfig):
            return HFReader(reader_config, specs, batch_size)

        if isinstance(reader_config, ImageFolderReaderConfig):
            return ImageFolderReader(reader_config, specs, batch_size)

        if isinstance(reader_config, PTReaderConfig):
            return _get_torchvision_reader(reader_config, specs, batch_size)

        if isinstance(reader_config, TFReaderConfig):
            return TFReader(reader_config, specs, batch_size)

        # if isinstance(reader_config, CSVReaderConfig):
        #     return CSVReader(reader_config, specs, batch_size)

        if isinstance(reader_config, VTABReaderConfig):
            return VTABReader(reader_config, specs, batch_size)

        raise RuntimeError(f"Unknown config {reader_config!r}")
