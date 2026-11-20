import numpy as np
import pytest

from capture.core.views import HemisphereSpec, azimuths_deg, hemisphere_views, serpentine


def spec(**kw) -> HemisphereSpec:
    base = dict(center=(0.0, 0.0, 0.1), radius=0.35, elevations_deg=(20.0, 50.0), views_per_ring=6)
    base.update(kw)
    return HemisphereSpec(**base)


def test_every_view_looks_at_centre_at_radius():
    s = spec()
    center = np.array(s.center)
    views = hemisphere_views(s)
    assert len(views) == 12
    for T in views:
        to_center = center - T[:3, 3]
        assert np.isclose(np.linalg.norm(to_center), s.radius)
        assert np.allclose(T[:3, 2], to_center / s.radius)


def test_elevations():
    s = spec(serpentine=False)
    views = hemisphere_views(s)
    heights = [T[2, 3] - s.center[2] for T in views]
    assert np.allclose(heights[:6], s.radius * np.sin(np.radians(20)))
    assert np.allclose(heights[6:], s.radius * np.sin(np.radians(50)))


def test_serpentine_reverses_odd_rings():
    assert serpentine([[1, 2, 3], [4, 5, 6], [7, 8]]) == [1, 2, 3, 6, 5, 4, 7, 8]
    plain = hemisphere_views(spec(serpentine=False))
    snake = hemisphere_views(spec(serpentine=True))
    assert np.allclose(snake[6], plain[11])  # second ring starts where the first one ended


def test_full_circle_excludes_duplicate_and_arc_includes_ends():
    assert np.allclose(azimuths_deg(4, 0, 360), [0, 90, 180, 270])
    assert np.allclose(azimuths_deg(3, -60, 60), [-60, 0, 60])
    assert np.allclose(azimuths_deg(2, 0, 360, offset_deg=15), [15, 195])


def test_views_per_ring_per_elevation():
    assert len(hemisphere_views(spec(views_per_ring=(8, 4)))) == 12
    with pytest.raises(ValueError):
        spec(views_per_ring=(8, 4, 2))


def test_invalid_radius():
    with pytest.raises(ValueError):
        spec(radius=0)
