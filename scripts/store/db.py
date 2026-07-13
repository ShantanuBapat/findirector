"""Database connection helpers for the pgvector store.

Connection config comes from environment variables. In local dev these are
loaded from a gitignored .env file (see .env.example for the keys); in
CI/production they come from the platform's secret store. Nothing sensitive is
hardcoded.
"""

import os

from dotenv import load_dotenv

# Load .env into os.environ once, on import. Real environment variables always
# win over .env (load_dotenv does not override existing vars by default), so
# CI/production values take precedence when set.
load_dotenv()


def get_dsn() -> str:
    """Build the Postgres DSN from environment variables.

    Reads POSTGRES_USER / PASSWORD / DB / HOST / PORT. Raises a clear error if a
    required variable is missing, rather than silently connecting to the wrong
    place or leaking a default password.
    """
    required = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
                "POSTGRES_HOST", "POSTGRES_PORT"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Copy .env.example to .env and fill it in."
        )

    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    host = os.environ["POSTGRES_HOST"]
    port = os.environ["POSTGRES_PORT"]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"
