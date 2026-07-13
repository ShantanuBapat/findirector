import psycopg

DSN = "postgresql://postgres:findirector@localhost:5432/findirector"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id                bigserial PRIMARY KEY,
    ticker            text    NOT NULL,
    fiscal_year       int     NOT NULL,
    filing_type       text    NOT NULL,
    accession_number  text    NOT NULL,
    section           text,
    content_type      text    NOT NULL,
    chunk_index       int     NOT NULL,
    n_tokens          int     NOT NULL,
    text              text    NOT NULL,
    embedding         vector(1024)
);
"""

with psycopg.connect(DSN) as conn:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()
        print("chunks table created")

        # Show the resulting column layout to confirm.
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'chunks'
            ORDER BY ordinal_position;
        """)
        for name, dtype in cur.fetchall():
            print(f"  {name:20} {dtype}")
