# Copyright © 2026 by Northwestern University. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""Tests for the episodic experiment-memory store."""

from __future__ import annotations

import pytest

from nuance_mcp.core import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_start_log_finish_roundtrip(store: MemoryStore):
    exp_id = store.start_experiment(
        "eels_survey",
        {"material": "TiO2", "core_loss_eV": "456"},
        material="TiO2",
        adapter_vendor="simulator",
        adapter_model="sim-1",
    )
    assert exp_id.startswith("exp-")

    store.log_tool_call(exp_id, "get_capabilities", {}, "eels present")
    store.log_tool_call(exp_id, "acquire_eels", {"energy_offset_eV": 456}, "ok")

    record = store.finish_experiment(exp_id, status="ok", summary="edges: Ti-L, O-K")

    assert record["status"] == "ok"
    assert record["material"] == "TiO2"
    assert record["arguments"]["core_loss_eV"] == "456"
    assert len(record["tool_calls"]) == 2
    assert record["tool_calls"][0]["tool_name"] == "get_capabilities"
    assert record["tool_calls"][1]["tool_name"] == "acquire_eels"
    assert record["finished_at"] is not None


def test_get_experiment_unknown_id_raises(store: MemoryStore):
    with pytest.raises(KeyError):
        store.get_experiment("exp-doesnotexist")


def test_recent_experiments_orders_newest_first(store: MemoryStore):
    first = store.start_experiment("beam_alignment", {})
    store.finish_experiment(first)
    second = store.start_experiment("hrtem_imaging", {"material": "Si"}, material="Si")
    store.finish_experiment(second)

    recent = store.recent_experiments(limit=10)
    assert [r["id"] for r in recent] == [second, first]


def test_recall_similar_filters_by_material_and_skill(store: MemoryStore):
    tio2_eels = store.start_experiment(
        "eels_survey", {"material": "TiO2"}, material="TiO2"
    )
    store.finish_experiment(tio2_eels)
    tio2_hrtem = store.start_experiment(
        "hrtem_imaging", {"material": "TiO2"}, material="TiO2"
    )
    store.finish_experiment(tio2_hrtem)
    si_eels = store.start_experiment("eels_survey", {"material": "Si"}, material="Si")
    store.finish_experiment(si_eels)

    by_material = store.recall_similar(material="TiO2")
    assert {r["id"] for r in by_material} == {tio2_eels, tio2_hrtem}

    by_both = store.recall_similar(material="TiO2", skill_name="eels_survey")
    assert [r["id"] for r in by_both] == [tio2_eels]

    assert store.recall_similar(material="GaN") == []


def test_store_persists_across_reconnect(tmp_path):
    db_path = str(tmp_path / "memory.sqlite3")

    store1 = MemoryStore(db_path)
    exp_id = store1.start_experiment("diffraction_survey", {"material": "Au"})
    store1.finish_experiment(exp_id, summary="fcc match")
    store1.close()

    store2 = MemoryStore(db_path)
    record = store2.get_experiment(exp_id)
    assert record["summary"] == "fcc match"
    store2.close()
