from enum import Enum

__all__ = ["Device", "DataType"]


class Device(Enum):
    """Device to use (whether to use hardware acceleration or not)."""

    GPU = "GPU"
    CPU = "CPU"


class DataType(Enum):
    """Specifies the type of the data."""

    IMAGE = "image"
    TEXT = "text"

    # Denotes a type that is other than image or text
    OTHER = "other"

    # Denotes a type that is not known - can be image, text or something else
    UNKNOWN = "unknown"

    @property
    def is_embed_type(self) -> bool:
        """Specifies whether the selected data type is suitable to be embedded.

        Returns:
            bool: True if the data type is suitable to be embedded or if the data type
            is unknown, False otherwise.
        """
        return self in {self.IMAGE, self.TEXT, self.UNKNOWN}
