from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.infra.minio_client import minio_available, upload_file  # noqa: E402


BUCKET = "maintainers-copilot-artifacts"
EVAL_REPORT_PATH = ROOT / "eval_report.json"


def timestamped_eval_report_key(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    timestamp = current.strftime("%Y/%m/%d/%H%M%S")
    return f"eval-reports/{timestamp}/eval_report.json"


def main() -> int:
    if not EVAL_REPORT_PATH.exists():
        print("eval_report.json is missing.", file=sys.stderr)
        return 1
    if not minio_available(timeout_seconds=2):
        print(
            "MinIO is not reachable. Start it with `docker compose up minio` "
            "or set MINIO_ENDPOINT/MINIO_ROOT_USER/MINIO_ROOT_PASSWORD.",
            file=sys.stderr,
        )
        return 1

    object_name = timestamped_eval_report_key()
    upload_file(BUCKET, object_name, EVAL_REPORT_PATH)
    print(f"Uploaded eval_report.json to s3://{BUCKET}/{object_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
