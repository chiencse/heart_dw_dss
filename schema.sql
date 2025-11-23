-- 0. Optional: create schema
CREATE SCHEMA IF NOT EXISTS heart_dw;

-- 1. Staging table (raw cleaned CSV/DF => load vào đây first)
CREATE TABLE IF NOT EXISTS heart_dw.staging_heart_raw (
  row_id SERIAL PRIMARY KEY,
  age INTEGER NOT NULL,
  sex TEXT NOT NULL,
  origin TEXT,               -- dataset / place of study
  cp TEXT,                   -- chest pain textual
  trestbps NUMERIC,          -- mm Hg
  chol NUMERIC,              -- mg/dl
  fbs BOOLEAN,               -- fasting blood sugar > 120
  restecg TEXT,
  thalach NUMERIC,
  exang BOOLEAN,
  oldpeak NUMERIC,
  slope TEXT,
  ca INTEGER,
  thal TEXT,
  num INTEGER,               -- target (label)
  event_time TIMESTAMP WITH TIME ZONE DEFAULT now(), -- fake or real timestamp
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 2. Dimensions (lookup tables)
CREATE TABLE IF NOT EXISTS heart_dw.dim_patient (
  patient_key BIGSERIAL PRIMARY KEY,
  unique_id TEXT UNIQUE,      -- optional mapping to original id
  age INTEGER NOT NULL,
  sex TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS heart_dw.dim_origin (
  origin_key SERIAL PRIMARY KEY,
  origin_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS heart_dw.dim_cp (
  cp_key SERIAL PRIMARY KEY,
  cp_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS heart_dw.dim_restecg (
  restecg_key SERIAL PRIMARY KEY,
  restecg_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS heart_dw.dim_slope (
  slope_key SERIAL PRIMARY KEY,
  slope_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS heart_dw.dim_thal (
  thal_key SERIAL PRIMARY KEY,
  thal_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS heart_dw.dim_date (
  date_key DATE PRIMARY KEY,
  year INTEGER NOT NULL,
  month INTEGER NOT NULL,
  day INTEGER NOT NULL,
  weekday INTEGER NOT NULL
);

-- 3. Fact table
CREATE TABLE IF NOT EXISTS heart_dw.fact_heart_assessment (
  fact_id BIGSERIAL PRIMARY KEY,
  patient_key BIGINT NOT NULL REFERENCES heart_dw.dim_patient(patient_key),
  date_key DATE NOT NULL REFERENCES heart_dw.dim_date(date_key),
  origin_key INT REFERENCES heart_dw.dim_origin(origin_key),
  cp_key INT REFERENCES heart_dw.dim_cp(cp_key),
  restecg_key INT REFERENCES heart_dw.dim_restecg(restecg_key),
  slope_key INT REFERENCES heart_dw.dim_slope(slope_key),
  thal_key INT REFERENCES heart_dw.dim_thal(thal_key),

  -- measures
  trestbps NUMERIC,
  chol NUMERIC,
  fbs BOOLEAN,
  thalach NUMERIC,
  exang BOOLEAN,
  oldpeak NUMERIC,
  ca INTEGER,
  target_num INTEGER, -- original 'num' label
  
  event_time TIMESTAMP WITH TIME ZONE DEFAULT now(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 4. Indexes to speed common queries
CREATE INDEX IF NOT EXISTS idx_fact_patient ON heart_dw.fact_heart_assessment(patient_key);
CREATE INDEX IF NOT EXISTS idx_fact_date ON heart_dw.fact_heart_assessment(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_target ON heart_dw.fact_heart_assessment(target_num);
