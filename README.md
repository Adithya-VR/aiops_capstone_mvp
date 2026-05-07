# AIOps Log Anomaly Detection Dashboard

This project is an AIOps prototype for parsing system/security logs, extracting
time-window features, scoring anomalous windows, generating alerts, clustering
similar alert windows, and visualizing the results in a Streamlit dashboard backed
by a FastAPI service.

The current implementation supports:

- BGL BlueGene/L supercomputer logs with ground-truth labels.
- OpenSSH authentication logs without ground-truth labels.
- Drain3 template mining.
- Sliding-window feature extraction.
- Isolation Forest anomaly scoring.
- Alert generation with severity tiers.
- TF-IDF + DBSCAN and MiniLM + DBSCAN alert clustering.
- FastAPI endpoints over processed Parquet artifacts.
- Streamlit dashboard with dataset, date, severity, and source-file filters.

## Project Structure

```text
.
├── api/
│   └── main.py                 # FastAPI service over processed artifacts
├── parsers/
│   ├── bgl.py                  # BGL raw log parser
│   └── openssh.py              # OpenSSH raw log parser
├── data/
│   ├── BGL.log                 # BGL source log, if available
│   └── OpenSSH/
│       └── SSH.log             # OpenSSH source log, if available
├── output/
│   ├── bgl/                    # BGL processed artifacts
│   └── openssh/                # OpenSSH processed artifacts
├── alert_generation.py         # Shared alert generation logic
├── alerts.py                   # TF-IDF alert clustering
├── alerts_minilm.py            # TF-IDF vs MiniLM clustering comparison
├── app.py                      # Streamlit dashboard
├── dataset_config.py           # Dataset configuration and artifact paths
├── pipeline.py                 # Parse, feature, model, score, metrics pipeline
├── smoke_test.py               # End-to-end artifact/API smoke checks
├── verify.py                   # Metrics/artifact verification
├── feature_verify.py           # Feature-window verification
├── requirements.txt
└── drain3.ini
```

## Datasets

### BGL

Default source path:

```text
data/BGL.log
```

Optional additional BGL log files can be placed under:

```text
data/BGL/*.log
```

BGL has ground-truth labels. The parser treats rows whose label field is not
`-` as anomalous.

Current configured windowing:

```text
Window size: 1 hour
Step size:   30 minutes
Mode:        sliding windows
```

### OpenSSH

Source path directory:

```text
data/OpenSSH/*.log
```

The OpenSSH parser processes all `.log` files in this directory, sorted by file
name. OpenSSH does not have ground-truth labels in this project, so evaluation is
unlabeled. The model flags the most unusual windows based on the configured
contamination value.

Current configured windowing:

```text
Window size: 5 minutes
Step size:   2.5 minutes
Mode:        sliding windows
Contamination: 0.03
```

OpenSSH uses `post_filter = predicted_only` because most OpenSSH levels are
security-relevant by design, making the generic evidence filter effectively
equivalent to raw model predictions.

## Multi-File Support

The pipeline now supports multiple raw log files per dataset.

For each parsed row, the pipeline records:

- `line_id`: global line id across all files.
- `source_file`: raw file path.
- `source_line_id`: line number within the source file.

The dashboard exposes source-file filters after artifacts are regenerated with
these fields.

Important: existing artifacts generated before this feature will still work, but
source-file filters will not appear until the pipeline is rerun.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Processing a Dataset

Run the full pipeline:

```powershell
python pipeline.py --dataset bgl
```

or:

```powershell
python pipeline.py --dataset openssh
```

The pipeline creates:

```text
output/<dataset>/parsed.parquet
output/<dataset>/features.parquet
output/<dataset>/scores.parquet
output/<dataset>/model.pkl
output/<dataset>/metrics.json
```

Generate TF-IDF alert clusters:

```powershell
python alerts.py --dataset bgl
python alerts.py --dataset openssh
```

Generate TF-IDF vs MiniLM clustering comparison:

```powershell
python alerts_minilm.py --dataset bgl
python alerts_minilm.py --dataset openssh
```

This creates:

```text
output/<dataset>/alerts.parquet
output/<dataset>/alerts_minilm.parquet
output/<dataset>/clustering_comparison.json
```

## Running the Application

Start the FastAPI service:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start the Streamlit dashboard in another terminal:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Dashboard Features

The dashboard includes:

- Dataset selector for BGL and OpenSSH.
- Overview metrics and score distribution.
- Log Explorer with pagination, level filters, search, and source-file filters.
- Anomaly Timeline with score threshold controls.
- Top predicted alert windows.
- Alert Clusters with date, severity, source-file, and clustering-method filters.
- TF-IDF vs MiniLM clustering comparison.

For unlabeled datasets such as OpenSSH, ground-truth metrics are hidden or marked
as unavailable. Predicted alert windows are shown instead.

## API Overview

Common dataset-scoped endpoints:

```text
GET /datasets
GET /datasets/{dataset}/stats
GET /datasets/{dataset}/logs
GET /datasets/{dataset}/source-files
GET /datasets/{dataset}/levels
GET /datasets/{dataset}/templates/top
GET /datasets/{dataset}/scores/timeline
GET /datasets/{dataset}/alerts
GET /datasets/{dataset}/alerts/summary
GET /datasets/{dataset}/clusters
GET /datasets/{dataset}/alerts/minilm/clusters
GET /datasets/{dataset}/clustering/comparison
GET /datasets/{dataset}/metrics
```

The API reads Parquet/JSON artifacts from `output/<dataset>/` and does not query
the raw logs directly.

## Artifact Flow

```text
Raw log files
    ↓
Parser + Drain3
    ↓
parsed.parquet
    ↓
Sliding-window feature extraction
    ↓
features.parquet
    ↓
Isolation Forest scoring
    ↓
scores.parquet + metrics.json
    ↓
Alert generation
    ↓
alerts.parquet
    ↓
TF-IDF / MiniLM clustering
    ↓
alerts_minilm.parquet + clustering_comparison.json
    ↓
FastAPI
    ↓
Streamlit dashboard
```

## Verification

Run unit/integration tests:

```powershell
python -m unittest discover -s tests
```

Run smoke tests:

```powershell
python smoke_test.py
```

Run metrics verification:

```powershell
python verify.py --dataset bgl
python verify.py --dataset openssh
```

Run feature-window verification:

```powershell
python feature_verify.py --dataset bgl
python feature_verify.py --dataset openssh
```

The `tests/` suite covers:

- BGL and OpenSSH parser behavior.
- Shared alert generation and severity assignment.
- Dataset configuration and project-root path anchoring.
- FastAPI health, dataset, log, metrics, and cluster endpoints.
- Processed artifact integrity and feature-window configuration.

## Notes on Alert Clustering

Alert clustering groups predicted anomalous windows by representative log message
or template similarity.

Current clustering methods:

- `TF-IDF + DBSCAN`: token-overlap similarity.
- `MiniLM + DBSCAN`: semantic embedding similarity.

The same error pattern can be clustered together even if it happens on different
dates. Date filtering in the dashboard narrows which alert windows are shown; it
does not retrain or recluster the model from scratch.

## Known Limitations

- OpenSSH has no ground-truth labels, so F1/accuracy are not meaningful for it.
- MiniLM model loading may download model files on first run.
- DBSCAN eps values are configured per vector space and should be tuned further
  for a more rigorous comparison.
- Existing artifacts must be regenerated to include newly added `source_file` and
  `top_level` fields.
- The project is currently designed for local analysis, not public multi-user
  deployment.

## Regenerating Everything

For BGL:

```powershell
python pipeline.py --dataset bgl
python alerts.py --dataset bgl
python alerts_minilm.py --dataset bgl
```

For OpenSSH:

```powershell
python pipeline.py --dataset openssh
python alerts.py --dataset openssh
python alerts_minilm.py --dataset openssh
```

If artifacts already exist, the pipeline may skip stages. Delete the relevant
files under `output/<dataset>/` to force regeneration.
