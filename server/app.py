# Copyright (c) 2026. ExecuCode Environment.

"""
FastAPI application entry point for ExecuCode environment.
"""

import os
from openenv.core.env_server.http_server import create_app

try:
    # Try package-style import
    from .environment import ExecuCodeEnvironment
    from ..models import ExecuCodeAction, ExecuCodeObservation
except ImportError:
    # Fallback to local import if not running as package
    from environment import ExecuCodeEnvironment
    from models import ExecuCodeAction, ExecuCodeObservation

# Create the FastAPI app
# We pass the class (factory) so the server can create a new instance per session
app = create_app(
    env_factory=ExecuCodeEnvironment,
    action_cls=ExecuCodeAction,
    observation_cls=ExecuCodeObservation,
    env_name="execucode",
)

if __name__ == "__main__":
    import uvicorn
    # Allow running directly for local development
    uvicorn.run(app, host="0.0.0.0", port=8000)
