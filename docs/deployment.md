# BEET 2.0 — Deployment Guide

This guide covers running BEET in three modes: local CLI, embedded web GUI,
and the REST API (containerised).

## Local CLI

Installation:

```bash
pip install -e .
```

Analyze a single text:

```bash
beet analyze --file submission.txt --profile screening
```

Batch analysis:

```bash
beet batch submissions.csv --text-column text --id-column submission_id \
    --cross-similarity --output-file results.csv
```

Evaluation, ablation, robustness, fairness:

```bash
beet eval data/dataset.jsonl --profile default
beet robustness data/dataset.jsonl --attacks all --confirm
beet fairness data/dataset.jsonl --group-by source --threshold 0.5
```

## Embedded GUI

No external dependencies beyond the base install:

```bash
beet gui --port 8877
```

The server is single-threaded and suitable for local/desktop use. For
multi-user deployments, front it with a reverse proxy or use the REST API.

## REST API (containerised)

Install API extras locally, or use the provided Dockerfile.

```bash
pip install -e ".[api]"
beet serve --host 0.0.0.0 --port 8000 --profile production
```

Docker:

```bash
docker compose up --build
```

Endpoints:

- `GET  /health`   — liveness + model info
- `GET  /config`   — current (redacted) config
- `POST /analyze`  — `{text, task_metadata?, submission_id?}`
- `POST /batch`    — `{items: [{id, text}], cross_similarity?: bool}`

## Production configuration

`configs/production.yaml` points at `/app/models/` for trained artifacts and
`/data/` for monitoring + privacy vault. Mount these directories into the
container:

```yaml
volumes:
  - ./models:/app/models:ro
  - beet-data:/data
```

## Retention purges

The privacy vault auto-purge runs on-demand via the CLI. Schedule via cron
or systemd timer on the container host:

```bash
beet privacy purge --vault-dir /data/vault --retention-days 90 --confirm
```

## Audit-chain validation

After a production incident or routine audit:

```bash
beet audit-validate /data/provenance/audit.jsonl
```

Exit code 0 = valid chain; 3 = tamper detected (errors to stderr).

## Model artifacts

See `scripts/` for training:

- `scripts/calibrate_detectors.py` — fit per-detector isotonic calibration
- `scripts/train_fusion.py` — train the EBM fusion model
- `scripts/calibrate_conformal.py` — calibrate conformal threshold

Outputs land in `models/` and are referenced by `fusion.model_path` etc. in
the active config. Execution requires a labeled JSONL dataset (see
`scripts/build_seed_dataset.py` — Phase 7).
