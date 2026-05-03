import pytest
from photo_scorer import ScoreResult, compute_total


def test_compute_total_weighted_average():
    result = ScoreResult(
        filename="IMG_001.jpg",
        technical=8.0,
        aesthetic=9.0,
        content=7.0,
        total=0.0,
        reason="test"
    )
    weights = {"technical": 0.30, "aesthetic": 0.40, "content": 0.30}
    total = compute_total(result, weights)
    # 8.0*0.30 + 9.0*0.40 + 7.0*0.30 = 2.4 + 3.6 + 2.1 = 8.1
    assert round(total, 2) == 8.1


def test_compute_total_equal_weights():
    result = ScoreResult(
        filename="IMG_002.jpg",
        technical=6.0,
        aesthetic=6.0,
        content=6.0,
        total=0.0,
        reason="test"
    )
    weights = {"technical": 0.33, "aesthetic": 0.34, "content": 0.33}
    total = compute_total(result, weights)
    assert round(total, 1) == 6.0
