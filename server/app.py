"""FastAPI application entry point for ExecuCode."""

from __future__ import annotations

try:
    from ..models import ExecuCodeAction, ExecuCodeObservation
    from .environment import ExecuCodeEnvironment
except ImportError:
    from models import ExecuCodeAction, ExecuCodeObservation
    from server.environment import ExecuCodeEnvironment

try:
    from openenv.core.env_server.http_server import create_app
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "openenv-core is required to serve ExecuCode. Install dependencies with "
        "`pip install -e .` before running the FastAPI app."
    ) from exc


app = create_app(
    ExecuCodeEnvironment,
    ExecuCodeAction,
    ExecuCodeObservation,
    env_name="execucode",
)
