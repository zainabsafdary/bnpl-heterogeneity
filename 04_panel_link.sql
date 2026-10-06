-- 04_panel_link.sql
-- Link people who answered both the 2024 and 2025 surveys.
-- Needs data/raw/public2024.csv (run 02_stack_years.sql first).
--
-- Why we think shedid links people across years: in the 2025 file, the first
-- four digits of shedid equal the year the person joined the survey, and
-- every 2024-cohort respondent carries a panel_weight. CONFIRM this in the
-- 2025 SHED codebook / documentation before relying on it.

CREATE OR REPLACE TABLE panel_2024_2025 AS
SELECT
    a.shedid,
    CAST(LEFT(a.shedid, 4) AS INTEGER)    AS entry_cohort,
    TRY_CAST(b.panel_weight AS DOUBLE)    AS panel_weight_2025,

    a.BNPL1       AS bnpl_2024,
    b.BNPL1       AS bnpl_2025,
    a.pay_casheqv AS cash400_2024,
    b.pay_casheqv AS cash400_2025,
    a.EF1         AS emerg3mo_2024,
    b.EF1         AS emerg3mo_2025,
    a.B2          AS wellbeing_2024,
    b.B2          AS wellbeing_2025,

    -- Transitions that the panel design is built to study.
    (a.pay_casheqv = 'Yes' AND b.pay_casheqv = 'No') AS lost_cash_buffer,
    (a.BNPL1 = 'No'        AND b.BNPL1 = 'Yes')      AS started_bnpl
FROM raw_all AS a
JOIN raw_all AS b
  ON a.shedid = b.shedid
WHERE a.survey_year = 2024
  AND b.survey_year = 2025;

COPY panel_2024_2025 TO 'data/clean/panel_2024_2025.csv' (HEADER, DELIMITER ',');

-- Match diagnostics: compare matched count with 2025 panel_weight count.
-- If n_matched is far below n_2025_with_panel_weight, the ID assumption is wrong.
SELECT
    (SELECT COUNT(*) FROM panel_2024_2025)                          AS n_matched,
    (SELECT COUNT(*) FROM raw_all
      WHERE survey_year = 2025 AND panel_weight IS NOT NULL)        AS n_2025_with_panel_weight,
    (SELECT SUM(CASE WHEN started_bnpl THEN 1 ELSE 0 END)
       FROM panel_2024_2025)                                        AS n_started_bnpl,
    (SELECT SUM(CASE WHEN lost_cash_buffer THEN 1 ELSE 0 END)
       FROM panel_2024_2025)                                        AS n_lost_cash_buffer;
