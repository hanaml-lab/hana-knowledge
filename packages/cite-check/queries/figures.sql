-- Arithmetic cite-check recomputes when a source holds a table.
-- Loaded by cite_check.numbers.growth_sql; the Python fallback must agree with these macros.

CREATE OR REPLACE MACRO growth_pct(start_value, end_value) AS
    (end_value - start_value) / start_value * 100.0;

CREATE OR REPLACE MACRO share_pct(part, whole) AS
    part / whole * 100.0;

CREATE OR REPLACE MACRO within_tolerance(stated, expected, tol) AS
    abs(stated - expected) <= tol * greatest(abs(stated), abs(expected));

-- name: growth
-- SELECT growth_pct("2023", "2024") FROM read_csv_auto('sources/segments.csv') WHERE lower(segment) = 'cloud';
