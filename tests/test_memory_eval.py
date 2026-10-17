"""Test multi-borrower memory evaluation (Task D)."""
from __future__ import annotations

from eval.evaluator import eval_memory


def test_eval_memory_20_borrowers():
    res = eval_memory(n_borrowers=20)
    assert res["borrowers_tested"] == 20
    assert res["call1_mean"] == 3.0
    assert res["call1_min"] == 3
    assert res["call1_max"] == 3
    assert res["call2_mean"] == 1.0
    assert res["call2_min"] == 1
    assert res["call2_max"] == 1
    assert res["share_memory_aware"] == 1.0
    assert res["turns_saved"] == 2.0
    assert res["learned_resolution_path"] is not None
