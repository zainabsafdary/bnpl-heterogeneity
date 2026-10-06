-- 01_load_2025.sql
-- Load the 2025 SHED public file into DuckDB.
-- all_varchar = true keeps every answer as text exactly as the Fed coded it.
-- We cast numbers explicitly later, so nothing gets silently mis-typed.

CREATE OR REPLACE TABLE raw_2025 AS
SELECT *
FROM read_csv('data/raw/public2025.csv', header = true, all_varchar = true);

-- Quick sanity check: expect 12,934 rows and a weighted mean weight of ~1.
SELECT
    COUNT(*)                                  AS n_rows,
    ROUND(AVG(TRY_CAST(weight AS DOUBLE)), 4) AS mean_weight,
    SUM(CASE WHEN BNPL1 = 'Yes' THEN 1 ELSE 0 END) AS n_bnpl_users
FROM raw_2025;
