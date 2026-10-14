# Copyright © 2026 by Northwestern University. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""Plugins: installed packages add tools and acquisition hooks to the server.

A package declares an entry point in the ``nuance_mcp.plugins`` group::

    [project.entry-points."nuance_mcp.plugins"]
    nuscope = "nuscope.mcp:register"

and ``register(ctx)`` receives a :class:`PluginContext`. With it a plugin can

* add MCP tools with ``@ctx.mcp.tool(...)`` (prefix their names, e.g.
  ``nuscope_verify``), and
* run code after every acquisition with ``@ctx.on_acquisition``: the hook
  gets the tool name, its validated parameters, the adapter's result object
  (:class:`~nuance_mcp.core.adapter.ImageReturn` or
  :class:`~nuance_mcp.core.adapter.SpectrumReturn`) and the instrument, and may
  return a dict that is added to the tool's response under its own key. This
  is how NUScope saves every acquisition as a sealed, signed file.

Plugins are loaded by default. ``NUANCE_MCP_PLUGINS=none`` disables them,
``NUANCE_MCP_PLUGINS=nuscope,tomo`` loads only those, and
``nuance-mcp serve --no-plugins`` does the same as ``none``. A plugin that
fails to load is reported by the ``list_plugins`` tool and never stops the
server. Nothing is printed to stdout, which carries the stdio MCP transport.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Callable, Optional

GROUP = "nuance_mcp.plugins"

AcquisitionHook = Callable[..., Optional[dict]]


@dataclass
class PluginContext:
    """What a plugin's ``register(ctx)`` receives."""

    mcp: Any                      # the FastMCP server
    adapter: Any                  # the bound MicroscopeAdapter
    hooks: dict[str, AcquisitionHook] = field(default_factory=dict)
    loaded: dict[str, str] = field(default_factory=dict)    # plugin name -> package version
    errors: dict[str, str] = field(default_factory=dict)    # plugin name -> error
    _current: str = "plugin"

    def on_acquisition(self, fn: AcquisitionHook) -> AcquisitionHook:
        """Register ``fn(tool=, params=, result=, instrument=) -> dict | None`` for every acquisition."""
        self.hooks[self._current] = fn
        return fn

    def instrument(self) -> dict:
        """Vendor, model and, when the adapter can tell, the column state at this moment."""
        info: dict[str, Any] = {"vendor": getattr(self.adapter, "vendor", None),
                                "model": getattr(self.adapter, "model", None)}
        try:
            state = self.adapter.get_state()
            info["state"] = {k: v for k, v in vars(state).items() if v not in (None, {}, [])}
        except Exception:  # a busy or offline column must not block the acquisition result
            pass
        return info


def acquisition_extras(ctx: Optional[PluginContext], tool: str, params: Any, result: Any) -> dict:
    """Run the acquisition hooks; returns ``{plugin name: hook output}`` (errors are reported, not raised)."""
    if ctx is None or not ctx.hooks:
        return {}
    if hasattr(params, "model_dump"):
        params = params.model_dump(exclude_none=True)
    instrument = ctx.instrument()
    out: dict[str, Any] = {}
    for name, hook in ctx.hooks.items():
        try:
            extra = hook(tool=tool, params=params or {}, result=result, instrument=instrument)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
        else:
            if extra:
                out[name] = extra
    return out


def _selected() -> Optional[set[str]]:
    value = os.environ.get("NUANCE_MCP_PLUGINS", "").strip()
    if not value:
        return None                       # all installed plugins
    if value.lower() in ("none", "0", "off", "false"):
        return set()
    return {v.strip() for v in value.split(",") if v.strip()}


def load_plugins(ctx: PluginContext) -> PluginContext:
    """Load the installed plugins into ``ctx`` and add the ``list_plugins`` tool."""
    wanted = _selected()
    for ep in entry_points(group=GROUP):
        if wanted is not None and ep.name not in wanted:
            continue
        ctx._current = ep.name
        try:
            ep.load()(ctx)
            dist = getattr(ep, "dist", None)
            ctx.loaded[ep.name] = getattr(dist, "version", "?") if dist else "?"
        except Exception as e:
            ctx.errors[ep.name] = f"{type(e).__name__}: {e}"
            print(f"nuance-mcp: plugin {ep.name!r} failed to load: {e}", file=sys.stderr, flush=True)
    ctx._current = "plugin"

    @ctx.mcp.tool(name="list_plugins")
    def list_plugins() -> dict:
        """Plugins loaded into this server (with versions), those that failed, and which run after acquisitions."""
        return {"loaded": ctx.loaded, "failed": ctx.errors, "acquisition_hooks": sorted(ctx.hooks)}

    return ctx


__all__ = ["GROUP", "PluginContext", "load_plugins", "acquisition_extras"]
