import psycopg

DSN = "postgresql://postgres:findirector@localhost:5432/findirector"

with psycopg.connect(DSN) as conn:
    with conn.cursor() as cur:
        # Prove the connection: ask Postgres its version.
        cur.execute("SELECT version();")
        print("connected:", cur.fetchone()[0][:60])

        # Enable pgvector in this database (idempotent).
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("pgvector extension enabled")

        # Confirm the extension is now registered, and its version.
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        print("pgvector version:", cur.fetchone()[0])
