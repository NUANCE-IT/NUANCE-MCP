# Copyright © 2026 by Northwestern University. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""Manual smoke test for the experiment-memory module.

Drives the real FastMCP server (simulator adapter) through the same
start_experiment / log_tool_call / finish_experiment / recall_similar_experiments
tool sequence an agent would follow, then reads it back both via the MCP
resource and directly off disk with the stdlib sqlite3 CLI.

Usage:
    python scripts/demo_memory.py                # persists to the default
                                                   # ~/.nuance_mcp/memory.sqlite3
    NUANCE_MEMORY_DB=:memory: python scripts/demo_memory.py   # ephemeral

Run it twice in a row (without :memory:) to see the second run's
recall_similar_experiments pick up the first run's record — that's the
persistence-across-restarts property that in-process JobRegistry doesn't have.
"""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client

from nuance_mcp.server import build_server
from nuance_mcp.core.memory import default_db_path


async def main() -> None:
    print(f"memory db: {default_db_path()}\n")

    mcp = build_server("simulator")
    async with Client(mcp) as client:
        started = await client.call_tool(
            "start_experiment",
            {
                "skill_name": "eels_survey",
                "arguments": {"material": "TiO2", "core_loss_eV": "456"},
                "material": "TiO2",
                "adapter_vendor": "simulator",
            },
        )
        exp_id = started.data["experiment_id"]
        print(f"started experiment: {exp_id}")

        await client.call_tool(
            "log_tool_call",
            {
                "experiment_id": exp_id,
                "tool_name": "get_capabilities",
                "result_summary": "eels present",
            },
        )
        await client.call_tool(
            "log_tool_call",
            {
                "experiment_id": exp_id,
                "tool_name": "acquire_eels",
                "arguments": {"energy_offset_eV": 456},
                "result_summary": "ok",
            },
        )
        finished = await client.call_tool(
            "finish_experiment",
            {"experiment_id": exp_id, "status": "ok", "summary": "Ti-L, O-K edges found"},
        )
        print("finished:", json.dumps(finished.data, indent=2))

        recalled = await client.call_tool(
            "recall_similar_experiments", {"material": "TiO2"}
        )
        print(f"\nrecall_similar_experiments(material='TiO2') -> "
              f"{len(recalled.data)} record(s):")
        for rec in recalled.data:
            print(f"  {rec['id']}  {rec['skill_name']}  {rec['summary']}")

        recent_blocks = await client.read_resource("memory://experiments/recent")
        recent = json.loads(recent_blocks[0].text)
        print(f"\nmemory://experiments/recent -> {len(recent)} record(s) total")


if __name__ == "__main__":
    asyncio.run(main())
