from schemas.requests.reader import Feature

from .._base import get_extraction_fn


def test_get_extraction_fn():
    data = {"first": {"second": "result1"}, "third": "result2"}
    fn = get_extraction_fn(
        [
            Feature(store_name="1", path=["first", "second"]),
            Feature(store_name="2", path=["third"]),
        ]
    )
    assert fn(data) == {"1": "result1", "2": "result2"}
