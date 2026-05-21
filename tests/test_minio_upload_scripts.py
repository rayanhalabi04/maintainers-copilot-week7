from datetime import UTC, datetime

from scripts import upload_eval_report_to_minio


def test_eval_report_upload_key_is_timestamped():
    key = upload_eval_report_to_minio.timestamped_eval_report_key(
        datetime(2026, 5, 21, 12, 34, 56, tzinfo=UTC)
    )

    assert key == "eval-reports/2026/05/21/123456/eval_report.json"


def test_eval_report_upload_uses_minio_client(monkeypatch, tmp_path):
    report = tmp_path / "eval_report.json"
    report.write_text("{}", encoding="utf-8")
    uploaded = {}

    monkeypatch.setattr(upload_eval_report_to_minio, "EVAL_REPORT_PATH", report)
    monkeypatch.setattr(upload_eval_report_to_minio, "minio_available", lambda timeout_seconds=2: True)
    monkeypatch.setattr(
        upload_eval_report_to_minio,
        "upload_file",
        lambda bucket, object_name, path: uploaded.update(
            {"bucket": bucket, "object_name": object_name, "path": path}
        ),
    )
    monkeypatch.setattr(
        upload_eval_report_to_minio,
        "timestamped_eval_report_key",
        lambda: "eval-reports/test/eval_report.json",
    )

    assert upload_eval_report_to_minio.main() == 0
    assert uploaded == {
        "bucket": "maintainers-copilot-artifacts",
        "object_name": "eval-reports/test/eval_report.json",
        "path": report,
    }
