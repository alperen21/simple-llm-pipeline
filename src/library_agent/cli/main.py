"""Root Typer application for the ``library-agent`` console script.

This module currently exports a help-only app. Use ``library-train`` and
``library-deploy`` for implemented command groups, or run
``python -m library_agent.agent.graph`` and ``python -m library_agent.mcp.server``
for the live assistant.
"""

import typer

app = typer.Typer(
    help=(
        "Library agent pipeline. Training: library-train. "
        "Packaging: library-deploy. Runtime: python -m library_agent.mcp.server "
        "and python -m library_agent.agent.graph."
    ),
)
