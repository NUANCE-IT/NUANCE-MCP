# Copyright © 2026 by Northwestern University. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""NUANCE-MCP — vendor-agnostic MCP server for multimodal EM."""

__version__ = "0.2.0"

from .core import (
    MicroscopeAdapter,
    MicroscopeState,
    ImageReturn,
    SpectrumReturn,
    CapabilityUnavailable,
    Capability,
    SimulatorAdapter,
    register_skills,
)
from .server import build_server, run_server

__all__ = [
    "__version__",
    "MicroscopeAdapter",
    "MicroscopeState",
    "ImageReturn",
    "SpectrumReturn",
    "CapabilityUnavailable",
    "Capability",
    "SimulatorAdapter",
    "register_skills",
    "build_server",
    "run_server",
]
