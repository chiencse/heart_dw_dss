# create_marts.py
"""
Create and populate 3 Data Marts in schema heart_dw:
 - mart_patient_profile
 - mart_clinical_event_flat
 - mart_time_trend_monthly

Usage:
  - Set DATABASE_URL environment variable (preferred) or replace DATABASE_URL below.
  - pip install sqlalchemy psycopg2-binary
  - python create_marts.py
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

# -------- CONFIG --------
# Option A: read from env
DATABASE_URL = os.environ.get("POSTGRES_CONNECTION_STRING") or \
    "postgresql://neondb_owner:npg_UErD4X9VCZWl@ep-long-hat-a17op69l-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# If using in Colab where you previously created engine via "postgresql://" (no +psycopg2),
# the SQLAlchemy URL must carry a DBAPI; above uses +psycopg2 which is typical.
# ------------------------

def create_engine_safe(db_url: str):
    # Validate URL and create engine
    u = make_url(db_url)
    # ensure driver name contains psycopg2 for COPY/fast ops (optional)
    engine = create_engine(db_url, echo=False, future=True)
    return engine

DDL_CREATE_MARTS = """
-- 1) mart_patient_profile
CREATE TABLE IF NOT EXISTS heart_dw.mart_patient_profile (
  patient_key BIGINT PRIMARY KEY,
  unique_id TEXT,
  age INTEGER,
  age_group TEXT,
  sex TEXT,
  primary_origin TEXT,
  total_events BIGINT,
  first_event_date DATE,
  last_event_date DATE,
  avg_chol NUMERIC,
  avg_trestbps NUMERIC,
  avg_thalach NUMERIC,
  max_oldpeak NUMERIC,
  ever_exang BOOLEAN,
  percent_positive NUMERIC,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 2) mart_clinical_event_flat
CREATE TABLE IF NOT EXISTS heart_dw.mart_clinical_event_flat (
  fact_id BIGINT PRIMARY KEY,
  patient_key BIGINT,
  unique_id TEXT,
  event_date DATE,
  event_time TIMESTAMP WITH TIME ZONE,
  age INTEGER,
  sex TEXT,
  origin TEXT,
  cp TEXT,
  restecg TEXT,
  slope TEXT,
  thal TEXT,
  trestbps NUMERIC,
  chol NUMERIC,
  fbs BOOLEAN,
  thalach NUMERIC,
  exang BOOLEAN,
  oldpeak NUMERIC,
  ca INTEGER,
  target_num INTEGER,
  age_bin TEXT,
  chol_cat TEXT,
  bp_stage TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 3) mart_time_trend_monthly
CREATE TABLE IF NOT EXISTS heart_dw.mart_time_trend_monthly (
  year INTEGER,
  month INTEGER,
  period DATE,
  total_events BIGINT,
  positive_events BIGINT,
  incidence_rate NUMERIC,
  avg_chol NUMERIC,
  avg_trestbps NUMERIC,
  avg_age NUMERIC,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  PRIMARY KEY (year, month)
);
"""

SQL_POPULATE_PATIENT_PROFILE = """
TRUNCATE TABLE heart_dw.mart_patient_profile;

INSERT INTO heart_dw.mart_patient_profile (
  patient_key, unique_id, age, age_group, sex, primary_origin,
  total_events, first_event_date, last_event_date,
  avg_chol, avg_trestbps, avg_thalach, max_oldpeak, ever_exang, percent_positive
)
SELECT
  p.patient_key,
  p.unique_id,
  p.age,
  CASE
    WHEN p.age < 40 THEN '<40'
    WHEN p.age BETWEEN 40 AND 54 THEN '40-54'
    WHEN p.age BETWEEN 55 AND 69 THEN '55-69'
    ELSE '70+'
  END AS age_group,
  p.sex,
  (
    SELECT o2.origin_name
    FROM heart_dw.fact_heart_assessment f2
    LEFT JOIN heart_dw.dim_origin o2 ON f2.origin_key = o2.origin_key
    WHERE f2.patient_key = p.patient_key
    GROUP BY o2.origin_name
    ORDER BY COUNT(*) DESC NULLS LAST
    LIMIT 1
  ) AS primary_origin,
  COUNT(f.fact_id) AS total_events,
  MIN(f.date_key) AS first_event_date,
  MAX(f.date_key) AS last_event_date,
  AVG(f.chol) AS avg_chol,
  AVG(f.trestbps) AS avg_trestbps,
  AVG(f.thalach) AS avg_thalach,
  MAX(f.oldpeak) AS max_oldpeak,
  BOOL_OR(f.exang) AS ever_exang,
  CASE WHEN COUNT(*) = 0 THEN 0
       ELSE SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*),0)
  END AS percent_positive
FROM heart_dw.dim_patient p
LEFT JOIN heart_dw.fact_heart_assessment f ON f.patient_key = p.patient_key
GROUP BY p.patient_key, p.unique_id, p.age, p.sex;
"""

SQL_POPULATE_EVENT_FLAT = """
TRUNCATE TABLE heart_dw.mart_clinical_event_flat;

INSERT INTO heart_dw.mart_clinical_event_flat (
  fact_id, patient_key, unique_id, event_date, event_time,
  age, sex, origin, cp, restecg, slope, thal,
  trestbps, chol, fbs, thalach, exang, oldpeak, ca, target_num,
  age_bin, chol_cat, bp_stage
)
SELECT
  f.fact_id,
  p.patient_key,
  p.unique_id,
  f.date_key AS event_date,
  f.event_time,
  p.age,
  p.sex,
  o.origin_name,
  cp.cp_name,
  r.restecg_name,
  sl.slope_name,
  th.thal_name,
  f.trestbps,
  f.chol,
  f.fbs,
  f.thalach,
  f.exang,
  f.oldpeak,
  f.ca,
  f.target_num,
  CASE
    WHEN p.age < 40 THEN '<40'
    WHEN p.age BETWEEN 40 AND 54 THEN '40-54'
    WHEN p.age BETWEEN 55 AND 69 THEN '55-69'
    ELSE '70+'
  END AS age_bin,
  CASE
    WHEN f.chol IS NULL THEN NULL
    WHEN f.chol < 200 THEN 'desirable'
    WHEN f.chol BETWEEN 200 AND 239 THEN 'borderline_high'
    ELSE 'high'
  END AS chol_cat,
  CASE
    WHEN f.trestbps IS NULL THEN NULL
    WHEN f.trestbps < 120 THEN 'normal'
    WHEN f.trestbps BETWEEN 120 AND 129 THEN 'elevated'
    WHEN f.trestbps BETWEEN 130 AND 139 THEN 'stage1'
    WHEN f.trestbps >= 140 THEN 'stage2'
    ELSE 'unknown'
  END AS bp_stage
FROM heart_dw.fact_heart_assessment f
LEFT JOIN heart_dw.dim_patient p ON f.patient_key = p.patient_key
LEFT JOIN heart_dw.dim_origin o ON f.origin_key = o.origin_key
LEFT JOIN heart_dw.dim_cp cp ON f.cp_key = cp.cp_key
LEFT JOIN heart_dw.dim_restecg r ON f.restecg_key = r.restecg_key
LEFT JOIN heart_dw.dim_slope sl ON f.slope_key = sl.slope_key
LEFT JOIN heart_dw.dim_thal th ON f.thal_key = th.thal_key;
"""

SQL_POPULATE_TIME_TREND = """
TRUNCATE TABLE heart_dw.mart_time_trend_monthly;

INSERT INTO heart_dw.mart_time_trend_monthly (
  year, month, period, total_events, positive_events, incidence_rate,
  avg_chol, avg_trestbps, avg_age
)
SELECT
  EXTRACT(YEAR FROM f.event_time)::int AS year,
  EXTRACT(MONTH FROM f.event_time)::int AS month,
  DATE_TRUNC('month', f.event_time)::date AS period,
  COUNT(*) AS total_events,
  SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::bigint AS positive_events,
  CASE WHEN COUNT(*) = 0 THEN 0
       ELSE SUM(CASE WHEN f.target_num > 0 THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*),0)
  END AS incidence_rate,
  AVG(f.chol) AS avg_chol,
  AVG(f.trestbps) AS avg_trestbps,
  AVG(p.age) AS avg_age
FROM heart_dw.fact_heart_assessment f
LEFT JOIN heart_dw.dim_patient p ON f.patient_key = p.patient_key
GROUP BY 1,2,3
ORDER BY 1,2;
"""

def run():
    engine = create_engine_safe(DATABASE_URL)
    print("Connecting to", DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL)

    with engine.begin() as conn:
        # Create the mart tables
        print("Creating mart tables (if not exists)...")
        conn.execute(text(DDL_CREATE_MARTS))
        print("Tables created.")

        # Populate patient profile
        print("Populating mart_patient_profile ...")
        conn.execute(text(SQL_POPULATE_PATIENT_PROFILE))
        print("mart_patient_profile populated.")

        # Populate event flat mart
        print("Populating mart_clinical_event_flat ...")
        conn.execute(text(SQL_POPULATE_EVENT_FLAT))
        print("mart_clinical_event_flat populated.")

        # Populate time trend mart
        print("Populating mart_time_trend_monthly ...")
        conn.execute(text(SQL_POPULATE_TIME_TREND))
        print("mart_time_trend_monthly populated.")

    print("All done. Marts created and populated.")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("Error:", e)
        raise
