-- 03_bnpl_lca_input.sql
-- Build the analysis table for the latent class model: one row per BNPL user
-- in 2025, with each indicator recoded into a small number of clear groups.
--
-- Design choices (explain these on your website):
--   * Indicators describe WHO the user is (cash buffer, credit habits, income
--     stability, wellbeing, motive, risk attitude).
--   * The outcome (paid_late) is deliberately NOT an indicator. It is kept as a
--     "distal outcome" and compared across classes afterward, so the classes
--     are not defined by the very thing we want to explain.
--   * VERIFY in the SHED 2025 codebook: BNPL3 is treated here as "paid late".

CREATE OR REPLACE TABLE bnpl_lca_input AS
WITH dur AS (
    SELECT MEDIAN(TRY_CAST(duration AS DOUBLE)) AS median_duration
    FROM raw_2025
)
SELECT
    r.shedid,
    TRY_CAST(r.weight AS DOUBLE) AS weight,

    -- ---------- Latent class indicators ----------
    r.pay_casheqv AS cash_400,          -- covers a $400 expense with cash/equivalent
    r.EF1         AS emerg_3mo,         -- has 3 months of emergency savings

    CASE
        WHEN r.C2A = 'No'                         THEN 'No credit card'
        WHEN r.C4A LIKE 'Never%'                  THEN 'Never revolves'
        WHEN r.C4A IN ('Once', 'Some of the time') THEN 'Sometimes revolves'
        WHEN r.C4A = 'Most or all of the time'    THEN 'Usually revolves'
    END AS cc_revolve,

    CASE r.I9
        WHEN 'Roughly the same amount each month'      THEN 'Stable'
        WHEN 'Occasionally varies from month to month' THEN 'Occasionally varies'
        WHEN 'Varies quite often from month to month'  THEN 'Often varies'
    END AS income_vol,

    r.B2 AS wellbeing,                  -- 4-level self-rated financial wellbeing

    CASE
        WHEN r.BNPL5 IN ('Only way I could afford it',
                         'Only accepted payment method I had')       THEN 'Afford / no other option'
        WHEN r.BNPL5 IN ('Wanted to spread out payments',
                         'Wanted a fixed number of payments')        THEN 'Spread payments'
        WHEN r.BNPL5 = 'Avoid interest charges'                      THEN 'Avoid interest'
        WHEN r.BNPL5 = 'Convenience'                                 THEN 'Convenience'
        WHEN r.BNPL5 = 'Did not want to use a credit card'           THEN 'Avoid credit card'
    END AS bnpl_reason,

    CASE
        WHEN TRY_CAST(regexp_extract(r.FL0, '^(\d+)', 1) AS INTEGER) <= 3 THEN 'Low (0-3)'
        WHEN TRY_CAST(regexp_extract(r.FL0, '^(\d+)', 1) AS INTEGER) <= 6 THEN 'Medium (4-6)'
        WHEN TRY_CAST(regexp_extract(r.FL0, '^(\d+)', 1) AS INTEGER) <= 10 THEN 'High (7-10)'
    END AS risk_tol,

    -- ---------- Distal outcome ----------
    CASE r.BNPL3 WHEN 'Yes' THEN 1 WHEN 'No' THEN 0 END AS paid_late,

    -- ---------- Covariates for later models ----------
    TRY_CAST(r.ppage AS INTEGER) AS age,
    r.inc_4cat_50k AS income_cat,
    r.educ_4cat    AS educ_cat,
    r.race_5cat    AS race_cat,
    r.malefemale   AS sex,
    r.ppstaten     AS state,

    -- ---------- Data quality flags ----------
    -- Speeder: finished in under half the median interview time.
    (TRY_CAST(r.duration AS DOUBLE) < 0.5 * dur.median_duration) AS speeder,
    -- Any key answer filled in by the Fed's imputation rather than reported.
    (r.BNPL1_iflag = '1' OR r.BNPL3_iflag = '1' OR r.BNPL5_iflag = '1'
     OR r.EF1_iflag = '1' OR r.C2A_iflag = '1' OR r.C4A_iflag = '1'
     OR r.I9_iflag  = '1' OR r.B2_iflag  = '1' OR r.FL0_iflag = '1') AS any_imputed

FROM raw_2025 AS r
CROSS JOIN dur
WHERE r.BNPL1 = 'Yes';

COPY bnpl_lca_input TO 'data/clean/bnpl_lca_input.csv' (HEADER, DELIMITER ',');

-- Summary printed when this file runs.
SELECT
    COUNT(*)                                     AS n_bnpl_users,
    ROUND(AVG(paid_late), 3)                     AS share_paid_late_unweighted,
    SUM(CASE WHEN speeder     THEN 1 ELSE 0 END) AS n_speeders,
    SUM(CASE WHEN any_imputed THEN 1 ELSE 0 END) AS n_any_imputed
FROM bnpl_lca_input;
