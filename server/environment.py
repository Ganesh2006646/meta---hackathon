"""Root-level environment.py — required by openenv validate.

Re-exports ExecuCodeEnvironment from its implementation module so that the
openenv validator can discover it at the package root as required by the spec.
"""

from execucode.server.environment import ExecuCodeEnvironment  # noqa: F401

__all__ = ["ExecuCodeEnvironment"]
