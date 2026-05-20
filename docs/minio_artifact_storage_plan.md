# MinIO Artifact Storage Plan

MinIO is not currently configured in this repo. There is no `docker-compose`
MinIO service, S3-compatible client dependency, artifact bucket configuration,
or upload script, so model artifact storage should remain marked not done until
those pieces exist.

## Docker Compose Service

Add a `minio` service to `docker-compose.yml` with:

- image: `minio/minio`
- command: `server /data --console-address ":9001"`
- ports: `9000:9000` and `9001:9001`
- volume: `minio_data:/data`
- environment variables for root credentials

## Required Environment Variables

- `MINIO_ENDPOINT`, for example `localhost:9000`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_SECURE`, usually `false` for local development
- `MODEL_ARTIFACT_BUCKET`, suggested value: `maintainers-copilot-artifacts`

## Artifacts To Upload

- `backend/model_server/artifacts/tfidf_logreg_baseline.joblib`
- `backend/model_server/artifacts/model_card.json`
- `backend/model_server/artifacts/artifact_manifest.json`
- `backend/model_server/artifacts/final_model_comparison.json`

## Upload Script To Add Later

Create `scripts/upload_model_artifacts_to_minio.py` after MinIO is configured.
It should:

- load the environment variables above
- create the artifact bucket if missing
- upload the four artifact files
- preserve relative object names under a model/version prefix
- fail clearly if `artifact_manifest.json` is missing

## Backend Verification Later

The backend should download or verify `artifact_manifest.json` before loading a
model artifact. It should compare the manifest SHA-256 values against local or
downloaded files and refuse to serve the model when a hash does not match.
