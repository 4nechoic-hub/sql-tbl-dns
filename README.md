# Turbulent Boundary Layer SQL Analytics Pipeline

An end-to-end data engineering and machine learning pipeline for turbulent boundary layer analysis, built on **real DNS data** from KTH Royal Institute of Technology.

## Overview

This project ingests wall-normal profiles of velocity statistics and turbulent kinetic energy budgets from Direct Numerical Simulation (DNS) of a zero-pressure-gradient turbulent boundary layer into a relational database, applies **SQL-driven feature engineering**, and trains ML models for:

- **Boundary layer region classification** — Identifying viscous sublayer, buffer layer, log layer, and wake regions using Random Forest (99.7% accuracy)
- **Re_tau regression** — Predicting friction Reynolds number from SQL-engineered turbulence features (R² = 0.88)
- **Anomaly detection** — Detecting unusual profile points via Isolation Forest and Local Outlier Factor

## Data Source

**KTH DNS Dataset** — Schlatter & Örlü, *J. Fluid Mech.* 659:116-126, 2010

- Spatially developing zero-pressure-gradient turbulent boundary layer
- Resolution: 8192 × 513 × 768 spectral modes
- 10 Reynolds numbers: Re_θ = 670 to 4060
- Variables: mean velocity, Reynolds stresses, vorticity, pressure, TKE budgets
- **30,780 real DNS data points** ingested into the database

Data available at: https://www.mech.kth.se/~pschlatt/DATA/

## Architecture

```
KTH DNS .prof files (velocity profiles + TKE budgets)
        │
        ▼
  Data Ingestion (Python parser)
        │
        ▼
  SQLite / PostgreSQL Database
  ┌─────────────────────────────────────┐
  │  simulation_conditions (10 Re_θ)    │
  │  velocity_profiles (5,130 rows)     │
  │  tke_budgets (5,130 rows)           │
  │  reynolds_stress_budgets (20,520)   │
  │  derived_features                   │
  │  predictions                        │
  │  model_registry                     │
  └─────────────────────────────────────┘
        │
        ▼
  SQL Feature Engineering
  (Window functions, CTEs, aggregations, JOINs)
        │
        ▼
  Python ML Pipeline
  (Random Forest, Gradient Boosting, Isolation Forest, LOF)
        │
        ▼
  Predictions → Written back to DB
```

## Key Results

| Task | Method | Metric |
|---|---|---|
| BL region classification | Random Forest | **99.7% accuracy** (5-fold CV: 98.3%) |
| Re_τ prediction | Gradient Boosting | **R² = 0.88** (LOO-CV) |
| Anomaly detection | Isolation Forest + LOF | 3% contamination threshold |

## SQL Skills Demonstrated

| SQL Concept | Where Used |
|---|---|
| `CREATE TABLE`, constraints, foreign keys | `sql/schema/01_create_tables.sql` |
| `SELECT`, `WHERE`, `ORDER BY`, `LIMIT` | `sql/queries/01_exploratory_analysis.sql` |
| `GROUP BY`, `HAVING`, aggregate functions | `sql/queries/01_exploratory_analysis.sql` |
| Window functions (`OVER`, `PARTITION BY`, `LAG`, `LEAD`) | `sql/queries/02_feature_engineering.sql` |
| `ROW_NUMBER`, `NTILE` | `sql/queries/02_feature_engineering.sql` |
| CTEs (`WITH ... AS`), chained CTEs | `sql/queries/02_feature_engineering.sql` |
| Correlated subqueries | `sql/queries/02_feature_engineering.sql` |
| `INNER JOIN`, `LEFT JOIN` | Throughout |
| `CASE` expressions | Region classification, categorisation |
| `INSERT`, `DELETE`, transactions (ACID) | `src/ml_pipeline.py` |
| `CREATE INDEX` | `sql/schema/01_create_tables.sql` |

## Quick Start

```bash
# Clone
git clone https://github.com/4nechoic-hub/tbl-sql-analytics.git
cd tbl-sql-analytics

# Install dependencies
pip install -r requirements.txt

# Ingest KTH DNS data into database
python src/ingest_kth.py --data-dir data/raw/kth_dns --backend sqlite

# Generate publication-quality figures (10 turbulence + 3 ML figures)
python src/generate_figures.py
python src/ml_pipeline.py --backend sqlite
```

## Figures

13 publication-quality figures generated from real DNS data:

**Turbulence Physics** (Figs 1–10): Mean velocity profiles (U+ vs y+), Reynolds stress profiles, TKE profiles (inner/outer scaling), TKE budget terms, integral parameter evolution (cf, Re_τ, H12), contour maps (Re_θ × y+), higher-order statistics (skewness, flatness), velocity defect profiles, vorticity RMS, 3D turbulence intensity surface.

**Machine Learning** (Figs 11–13): Region classification confusion matrix + feature importance, Re_τ regression (true vs predicted), anomaly detection spatial map.

## Technical Stack

- **Database:** PostgreSQL 16 / SQLite (portable)
- **Python:** pandas, scikit-learn, sqlalchemy, scipy, matplotlib
- **Data:** KTH DNS (Schlatter & Örlü, 2010)
- **Infrastructure:** Docker, GitHub Actions CI/CD

## Author

**Tingyi Zhang**
Postdoctoral Research Associate, UNSW Sydney
[GitHub](https://github.com/4nechoic-hub) · [LinkedIn](https://www.linkedin.com/in/tingyi-zhang-au/)

## References

Schlatter, P. and Örlü, R. (2010). Assessment of direct numerical simulation data of turbulent boundary layers. *J. Fluid Mech.*, 659:116-126.

Schlatter, P., Örlü, R., Li, Q., Brethouwer, G., Fransson, J.H.M., Johansson, A.V., Alfredsson, P.H. and Henningson, D.S. (2009). Turbulent boundary layers up to Re_θ=2500 studied through simulation and experiment. *Physics of Fluids*, 21, 051702.

## License

MIT License — see [LICENSE](LICENSE) for details.
