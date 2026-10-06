"""Run the project's SQL files in order against a local DuckDB database.

Usage (from the project root):
    python scripts/run_sql.py                 # runs sql/01 ... sql/03
    python scripts/run_sql.py --with-panel    # also stacks years and links the panel
    python scripts/run_sql.py sql/02_stack_years.sql   # run specific files

Each SQL file ends with a SELECT that summarizes what it built; that result
is printed so you can check the numbers every time you rebuild.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "shed.duckdb"

CORE = ["sql/01_load_2025.sql", "sql/03_bnpl_lca_input.sql"]
PANEL = ["sql/02_stack_years.sql", "sql/04_panel_link.sql"]


def run_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    print(f"\n=== {path.relative_to(ROOT)} ===")
    result = con.execute(path.read_text())  # runs all statements; returns the last
    try:
        df = result.fetchdf()
        if not df.empty:
            print(df.to_string(index=False))
    except duckdb.Error:
        pass  # last statement returned no rows (e.g. COPY)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Specific SQL files to run")
    parser.add_argument("--with-panel", action="store_true",
                        help="Also stack all years and build the 2024-2025 panel")
    args = parser.parse_args()

    (ROOT / "data" / "clean").mkdir(parents=True, exist_ok=True)
    files = args.files or (CORE + (PANEL if args.with_panel else []))

    # Relative paths inside the SQL (data/raw/...) resolve from the project root.
    os.chdir(ROOT)

    with duckdb.connect(str(DB_PATH)) as con:
        for f in files:
            run_file(con, ROOT / f)


if __name__ == "__main__":
    main()
