from datetime import date

from pydantic import BaseModel, Field, root_validator
from schemas._base import _DefaultConfig


class IntegerRange(BaseModel):
    min: int = Field(
        ..., title="Minimal value", description="Minimal value (inclusive)", example=224
    )
    max: int = Field(
        ..., title="Maximal value", description="Maximal value (inclusive)", example=600
    )

    @root_validator
    def valid_range(cls, values: dict) -> dict:
        if "min" in values and "max" in values:
            min_: int = values["min"]
            max_: int = values["max"]

            if min_ > max_:
                raise ValueError(f"Invalid range (min={min_} > max={max_})")
        return values

    class Config(_DefaultConfig):
        pass


class DateRange(BaseModel):
    min: date = Field(
        ...,
        title="Minimal date",
        description="Minimal date (inclusive) passed as the format described in "
        "https://docs.python.org/3/library/datetime.html#datetime.date.fromisoformat",
    )
    max: date = Field(
        ...,
        title="Maximal date",
        description="Maximal date (inclusive) passed as the format described in "
        "https://docs.python.org/3/library/datetime.html#datetime.date.fromisoformat",
    )

    @root_validator
    def valid_range(cls, values: dict) -> dict:
        if "min" in values and "max" in values:
            min_: date = values["min"]
            max_: date = values["max"]

            if min_ > max_:
                raise ValueError(f"Invalid range (min={min_} > max={max_})")
        return values

    class Config(_DefaultConfig):
        pass
