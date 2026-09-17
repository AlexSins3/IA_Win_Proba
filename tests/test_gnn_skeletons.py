"""Tests des schémas de squelette (P3)."""

from __future__ import annotations

import numpy as np

from kata_pipeline.gnn.skeletons import (
    available_schemas,
    get_schema,
)


def test_registry_contains_expected():
    names = available_schemas()
    assert "mediapipe_15" in names
    assert "mediapipe_33" in names
    assert "canonical_body" in names  # alias -> 15


def test_alias_points_to_15():
    assert get_schema("canonical_body").num_joints == 15
    assert get_schema("canonical_body_hands_feet").num_joints == 33


def test_unknown_schema_raises():
    try:
        get_schema("does_not_exist")
    except KeyError as exc:
        assert "inconnu" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("KeyError attendu")


def test_adjacency_symmetric_and_selfloops():
    s = get_schema("mediapipe_15")
    a = s.adjacency(add_self_loops=True)
    assert a.shape == (15, 15)
    assert np.allclose(a, a.T, atol=1e-6)
    assert np.all(np.diag(a) > 0)  # self-loops présents


def test_parents_root_is_self():
    s = get_schema("mediapipe_15")
    parents = s.parents()
    assert len(parents) == 15
    assert parents[s.root] == s.root
    # Tous les nœuds sont atteints (arbre couvrant connexe).
    assert all(0 <= p < 15 for p in parents)


def test_symmetry_pairs_distinct():
    s = get_schema("mediapipe_15")
    for a, b in s.symmetry:
        assert a != b
