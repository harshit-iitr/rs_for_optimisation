import numpy as np

from analysis.selectivity import selectivity


def test_shared_unit_scores_zero_and_full_participation():
    sel, index, pr, n_dead = selectivity(np.ones((1, 10)))
    assert np.isclose(index[0], 0.0)
    assert np.isclose(pr[0], 10.0)
    assert n_dead == 0


def test_task_specific_unit_scores_near_one_and_participation_one():
    resp = np.zeros((1, 10)); resp[0, 3] = 1.0
    sel, index, pr, _ = selectivity(resp)
    assert np.isclose(index[0], 0.9)
    assert np.isclose(pr[0], 1.0)


def test_normalisation_removes_overall_gain():
    """A loud unit and a quiet unit with the same shape must score identically:
    the map shows selectivity, not gain."""
    shape = np.array([1.0, 0.5, 0.25, 0.1])
    sel, index, pr, _ = selectivity(np.stack([shape, 100 * shape]))
    assert np.allclose(sel[0], sel[1])
    assert np.isclose(index[0], index[1])


def test_dead_units_are_dropped_not_nan():
    resp = np.zeros((3, 5)); resp[0] = 1.0; resp[2] = 1.0
    sel, index, pr, n_dead = selectivity(resp)
    assert n_dead == 1
    assert sel.shape[0] == 2
    assert np.isfinite(index).all() and np.isfinite(pr).all()
