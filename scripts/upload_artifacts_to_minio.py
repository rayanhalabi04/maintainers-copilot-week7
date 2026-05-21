from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.infra.minio_client import minio_available, upload_file  # noqa: E402


BUCKET = "maintainers-copilot-artifacts"
ARTIFACTS = [
    ROOT / "backend/model_server/artifacts/tfidf_logreg_baseline.joblib",
    ROOT / "backend/model_server/artifacts/model_card.json",
    ROOT / "backend/model_server/artifacts/artifact_manifest.json",
    ROOT / "backend/model_server/artifacts/final_model_comparison.json",
    ROOT / "eval_report.json",
]


def main() -> int:
    if not minio_available(timeout_seconds=2):
        print(
            "MinIO is not reachable. Start it with `docker compose up minio` "
            "or set MINIO_ENDPOINT/MINIO_ROOT_USER/MINIO_ROOT_PASSWORD.",
            file=sys.stderr,
        )
        return 1

    for artifact in ARTIFACTS:
        if not artifact.exists():
            print(f"Skipping missing artifact: {artifact}")
            continue
        object_name = artifact.relative_to(ROOT).as_posix()
        upload_file(BUCKET, object_name, artifact)
        print(f"Uploaded {object_name} to s3://{BUCKET}/{object_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
