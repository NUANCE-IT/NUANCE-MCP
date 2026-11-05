# Copyright © 2026 by Northwestern University. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""FastMCP server with vendor-agnostic adapter selection.

Usage::

    nuance-mcp serve --adapter gatan
    nuance-mcp serve --adapter jeol --mode offline
    nuance-mcp serve --adapter simulator         # default
    nuance-mcp serve --adapter gatan --transport http --port 8000

The server is a thin glue layer: it instantiates the chosen adapter,
registers the generic typed tools (:mod:`nuance_mcp.tools`) against it,
registers the vendor-portable skills (:mod:`nuance_mcp.core.skills`),
and starts the chosen MCP transport.
"""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from .core import register_skills, MicroscopeAdapter, MemoryStore, register_memory
from .tools import register_tools
from .adapters import load_adapter


INSTRUCTIONS_TEMPLATE = """\
NUANCE-MCP — vendor-agnostic LLM control of multimodal electron microscopy.

This server is bound to a {vendor} ({model}) adapter. The tool surface is
identical regardless of vendor; capability availability is exposed through
``get_capabilities``. Always call ``get_microscope_state`` first.

Capabilities declared by this adapter:
{caps}
"""


def build_server(
    adapter_name: str = "simulator",
    adapter_kwargs: Optional[dict] = None,
    memory_db_path: Optional[str] = None,
) -> FastMCP:
    """Instantiate the adapter, register tools + skills + memory, return the server.

    ``memory_db_path`` overrides the persistent experiment-memory location
    (default resolved by :func:`nuance_mcp.core.memory.default_db_path`,
    honouring ``NUANCE_MEMORY_DB``); pass ``":memory:"`` for an ephemeral,
    non-persistent store (used by tests).
    """

    adapter_cls = load_adapter(adapter_name)
    adapter: MicroscopeAdapter = adapter_cls(**(adapter_kwargs or {}))
    adapter.open()

    caps = ", ".join(sorted(c.value for c in adapter.capabilities)) or "(none)"
    mcp = FastMCP(
        name="nuance_mcp",
        instructions=INSTRUCTIONS_TEMPLATE.format(
            vendor=adapter.vendor,
            model=adapter.model,
            caps=caps,
        ),
    )

    register_tools(mcp, adapter)
    register_skills(mcp)
    register_memory(mcp, MemoryStore(memory_db_path))
    return mcp


def run_server(
    adapter_name: str = "simulator",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    adapter_kwargs: Optional[dict] = None,
    memory_db_path: Optional[str] = None,
) -> None:
    mcp = build_server(adapter_name, adapter_kwargs, memory_db_path)
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=host, port=port)
