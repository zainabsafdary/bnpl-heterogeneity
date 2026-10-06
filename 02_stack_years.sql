-- 02_stack_years.sql
-- Stack every SHED year you have downloaded into one long table.
--
-- Put each year's CSV in data/raw/ named publicYYYY.csv
-- (public2021.csv, public2022.csv, ..., public2025.csv).
--
-- union_by_name = true matches columns by NAME, not position. If a question
-- was not asked in some year, that year simply gets NULL for the column.
-- The year comes from the file name, so it works even if a file lacks a
-- `year` column.

CREATE OR REPLACE TABLE raw_all AS
SELECT
    CAST(regexp_extract(filename, 'public(\d{4})\.csv', 1) AS INTEGER) AS survey_year,
    * EXCLUDE (filename)
FROM read_csv(
    'data/raw/public20*.csv',
    header        = true,
    all_varchar   = true,
    union_by_name = true,
    filename      = true
);

-- Harmonization check: how many people answered each key question, by year?
-- A count of 0 means the question is missing or renamed that year.
-- Check the codebook before pooling any year where a count is 0.
CREATE OR REPLACE VIEW coverage_by_year AS
SELECT
    survey_year,
    COUNT(*)          AS n_respondents,
    COUNT(BNPL1)      AS n_bnpl1,
    COUNT(BNPL3)      AS n_bnpl3,
    COUNT(BNPL5)      AS n_bnpl5,
    COUNT(EF1)        AS n_ef1,
    COUNT(pay_casheqv) AS n_pay_casheqv,
    COUNT(C4A)        AS n_c4a,
    COUNT(FL0)        AS n_fl0
FROM raw_all
GROUP BY survey_year
ORDER BY survey_year;

SELECT * FROM coverage_by_year;
