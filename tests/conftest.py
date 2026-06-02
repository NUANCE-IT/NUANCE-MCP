# Copyright (c) 2026 Roberto dos Reis
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""Shared pytest fixtures."""

import os
import sys

# Ensure the in-repo src/ is importable for development without `pip install`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
