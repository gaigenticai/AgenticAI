"""
Environment validation helpers
"""
import os


def require_env(name: str) -> str:
    """Return environment variable value or raise RuntimeError with guidance."""
    val = os.getenv(name, "")
    if not val:
        raise RuntimeError(f"Required environment variable '{name}' is not set. Set it in the environment or .env file.")
    return val


def require_envs(names):
    """Ensure each name in names is present in environment."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return True


