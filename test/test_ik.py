import numpy as np
import pytest

from capture.core.ik import ik_seeds, select_joints


def test_first_seed_is_current_state_and_rest_within_limits():
    seeds = ik_seeds(["j1", "j2"], attempts=5, limit_rad=2.6, rng=np.random.default_rng(0))
    assert len(seeds) == 5 and seeds[0] is None
    for s in seeds[1:]:
        assert set(s) == {"j1", "j2"}
        assert all(-2.6 <= v <= 2.6 for v in s.values())


def test_seeds_need_an_attempt():
    with pytest.raises(ValueError):
        ik_seeds(["j1"], attempts=0, limit_rad=1.0, rng=np.random.default_rng(0))


def test_select_joints_drops_gripper_and_reports_missing():
    sol = {"j1": 0.1, "j2": 0.2, "gripper": 0.5}
    assert select_joints(sol, ["j1", "j2"]) == {"j1": 0.1, "j2": 0.2}
    with pytest.raises(KeyError, match="j3"):
        select_joints(sol, ["j1", "j3"])
