"""Basic shape validation tests for placeholder dataset builder."""

from src.preprocessing.build_dataset import build_dataset


def test_dataset_has_expected_dimensions() -> None:
    x, y = build_dataset()
    assert x.ndim == 3
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
