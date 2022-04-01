from typing import Callable, Optional, Sequence

from pipeline import DataType
from schemas.requests.reader import Feature

__all__ = ["get_extraction_fn", "check_paths"]


def get_extraction_fn(features: Sequence[Feature]) -> Callable:
    """Get function that will extract features from a dictionary.

    Args:
        features (Sequence[Feature]): Features to extract.

    Returns:
        Callable: A function that extracts the specified features from a dictionary.
    """
    extractions = [
        "".join(
            [f"'{feature.store_name}': x", *[f"['{part}']" for part in feature.path]]
        )
        for feature in features
    ]

    return eval(f"lambda x: {{{','.join(extractions)}}}")


def check_paths(
    check_path_fn: Callable[[Sequence[str]], DataType],
    embed_feature_path: Optional[Sequence[str]],
    label_feature_path: Optional[Sequence[str]],
    other_features_paths: Optional[Sequence[Sequence[str]]],
) -> DataType:
    """Checks the validity of feature paths and the validity of feature types selected
    with paths.

    Args:
        check_path_fn (Callable[[Sequence[str], DataType]): A function that given a path
            to the feature checks that the path is valid and returns the data type of
            the selected feature.
        embed_feature_path (Sequence[str], optional): Path that leads  to the embed
            feature.
        label_feature_path (Sequence[str], optional): Path that leads to the label
            feature.
        other_features_paths (Sequence[Sequence[str]], optional): Paths to other
            features.

    Returns:
        DataType: Type of the embed feature.
    """
    if label_feature_path is not None:
        check_path_fn(label_feature_path)

    if other_features_paths is not None:
        for other_feature_path in other_features_paths:
            check_path_fn(other_feature_path)

    if embed_feature_path is None:
        embed_inferred_type = DataType.UNKNOWN
    else:
        embed_inferred_type = check_path_fn(embed_feature_path)
        if not embed_inferred_type.is_embed_type:
            raise ValueError("Inference can only be run on images and texts!")

    return embed_inferred_type
