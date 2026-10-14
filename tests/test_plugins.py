# Copyright © 2026 by Northwestern University. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""Plugins: entry-point loading, acquisition hooks, and the NUScope / NUScope-TOMO plugins when installed."""

import asyncio
from types import SimpleNamespace

import pytest

from nuance_mcp import plugins as pl
from nuance_mcp.server import build_server


def call(mcp, name, args=None):
    from fastmcp import Client

    async def go():
        async with Client(mcp) as c:
            return (await c.call_tool(name, args or {})).structured_content

    return asyncio.run(go())


def fake_entry_points(*eps):
    return lambda group: [e for e in eps if group == pl.GROUP]


def ep(name, fn):
    return SimpleNamespace(name=name, load=lambda: fn, dist=SimpleNamespace(version="9.9"))


def test_hook_output_is_added_to_acquisitions(monkeypatch):
    seen = []

    def register(ctx):
        @ctx.on_acquisition
        def hook(tool, params, result, instrument):
            seen.append((tool, params, result.data.shape, instrument["vendor"]))
            return {"saved": f"/data/{tool}.h5"}

    monkeypatch.setattr(pl, "entry_points", fake_entry_points(ep("demo", register)))
    mcp = build_server("simulator")
    out = call(mcp, "acquire_stem", {"payload": {"width": 64, "height": 64}})
    assert out["plugins"] == {"demo": {"saved": "/data/acquire_stem.h5"}}
    assert seen[0][0] == "acquire_stem" and seen[0][1]["width"] == 64 and seen[0][3] == "simulator"
    assert call(mcp, "list_plugins")["loaded"] == {"demo": "9.9"}


def test_failing_plugin_does_not_stop_the_server(monkeypatch):
    def broken(ctx):
        raise RuntimeError("boom")

    def bad_hook(ctx):
        ctx.on_acquisition(lambda **kw: 1 / 0)

    monkeypatch.setattr(pl, "entry_points", fake_entry_points(ep("broken", broken), ep("bad", bad_hook)))
    mcp = build_server("simulator")
    assert "boom" in call(mcp, "list_plugins")["failed"]["broken"]
    out = call(mcp, "acquire_tem_image", {"payload": {}})
    assert out["shape"] and "ZeroDivisionError" in out["plugins"]["bad"]["error"]


def test_selection_and_opt_out(monkeypatch):
    calls = []
    monkeypatch.setattr(pl, "entry_points", fake_entry_points(ep("a", lambda c: calls.append("a")),
                                                              ep("b", lambda c: calls.append("b"))))
    monkeypatch.setenv("NUANCE_MCP_PLUGINS", "b")
    build_server("simulator")
    monkeypatch.setenv("NUANCE_MCP_PLUGINS", "none")
    build_server("simulator")
    monkeypatch.delenv("NUANCE_MCP_PLUGINS")
    build_server("simulator", plugins=False)
    assert calls == ["b"]


# ----------------------------------------------------------------------------- real plugins (optional)
def test_nuscope_saves_sealed_signed_acquisitions(tmp_path, monkeypatch):
    pytest.importorskip("nuscope.mcp")
    monkeypatch.setenv("NUANCE_MCP_PLUGINS", "nuscope")
    monkeypatch.setenv("NUSCOPE_ACQUISITION_DIR", str(tmp_path / "acq"))
    monkeypatch.setenv("EMDATABRIDGE_HOME", str(tmp_path / "id"))
    monkeypatch.setenv("NUSCOPE_HOME", str(tmp_path / "id"))
    from EMDataBridge import identity
    identity.create("Ada Lovelace", orcid="0000-0002-1825-0097")
    mcp = build_server("simulator")

    out = call(mcp, "acquire_stem", {"payload": {"width": 64, "height": 64, "dwell_us": 5}})
    saved = out["plugins"]["nuscope"]
    assert saved["integrity"] == "signed" and saved["signed_by"][0].startswith("Ada Lovelace")
    d = call(mcp, "nuscope_describe", {"path": saved["file"]})
    assert d["integrity"] == "signed" and d["axes"][-1]["units"] in ("nm", "px")
    assert any("simulator" in s.lower() or "Microscope" in s for s in d["summary"])
    h = call(mcp, "nuscope_history", {"path": saved["file"]})
    assert any("acquire.acquire_stem" in s and "dwell_us=5" in s for s in h["steps"])
    st = call(mcp, "nuscope_analyze", {"path": saved["file"], "op": "stats"})
    assert "result" in st or "mean" in str(st)
    den = call(mcp, "nuscope_analyze", {"path": saved["file"], "op": "denoise", "params": {"method": "gaussian"}})
    assert den["integrity"] == "signed" and den["history_steps"] >= 4
    assert call(mcp, "nuscope_verify", {"path": saved["file"]})["status"] == "signed"

    eels = call(mcp, "acquire_eels", {"payload": {"exposure_s": 0.1}})["plugins"]["nuscope"]
    assert call(mcp, "nuscope_describe", {"path": eels["file"]})["axes"][0]["units"] == "eV"
    assert len(call(mcp, "nuscope_list", {"folder": str(tmp_path / "acq" / sorted(
        p.name for p in (tmp_path / "acq").iterdir())[0])})["files"]) == 3   # stem, its denoised copy, eels


def test_tomo_plugin_tools(tmp_path, monkeypatch):
    pytest.importorskip("nuscope_tomo.mcp")
    import numpy as np
    from nuscope_tomo import Volume, VoxelSize
    monkeypatch.setenv("NUANCE_MCP_PLUGINS", "tomo")
    Volume(np.zeros((4, 8, 8), np.uint8), VoxelSize(20, 5, 5), name="raw").to_zarr(tmp_path / "proj" / "raw.zarr")
    (tmp_path / "proj" / "tables").mkdir()
    (tmp_path / "proj" / "tables" / "phases.csv").write_text("phase,fraction\npore,0.1\nsolid,0.9\n")
    mcp = build_server("simulator")
    p = call(mcp, "tomo_project", {"project": str(tmp_path / "proj")})
    assert p["volumes"][0]["voxel_nm"] == [20.0, 5.0, 5.0] and "tables/phases.csv" in p["tables"]
    t = call(mcp, "tomo_table", {"project": str(tmp_path / "proj"), "table": "tables/phases.csv"})
    assert t["rows"] == 2
    assert "outside" in str(call_safely(mcp, "tomo_table", {"project": str(tmp_path / "proj"), "table": "../../etc/hosts"}))
    assert len(call(mcp, "tomo_ops")) > 10


def call_safely(mcp, name, args):
    try:
        return call(mcp, name, args)
    except Exception as e:  # fastmcp raises ToolError for tool exceptions
        return e
