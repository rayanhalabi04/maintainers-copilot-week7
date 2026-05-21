import hashlib
import hmac
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


class MinioConfig(TypedDict):
    endpoint: str
    access_key: str
    secret_key: str
    region: str


def get_minio_config() -> MinioConfig:
    return {
        "endpoint": os.getenv("MINIO_ENDPOINT", "http://localhost:9000").rstrip("/"),
        "access_key": os.getenv("MINIO_ROOT_USER", "minioadmin"),
        "secret_key": os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        "region": os.getenv("MINIO_REGION", "us-east-1"),
    }


def minio_available(timeout_seconds: float = 1.0) -> bool:
    endpoint = get_minio_config()["endpoint"]
    try:
        with urlopen(f"{endpoint}/minio/health/live", timeout=timeout_seconds) as response:
            return 200 <= response.status < 500
    except (OSError, URLError):
        return False


def upload_file(bucket: str, object_name: str, path: str | Path) -> None:
    config = get_minio_config()
    file_path = Path(path)
    data = file_path.read_bytes()
    _put(bucket, "", b"", config)
    _put(bucket, object_name, data, config)


def _put(bucket: str, object_name: str, data: bytes, config: MinioConfig) -> None:
    endpoint = config["endpoint"]
    parsed = urlparse(endpoint)
    encoded_object = quote(object_name.strip("/"))
    path = f"/{bucket}" + (f"/{encoded_object}" if encoded_object else "")
    url = f"{endpoint}{path}"
    headers = _signed_headers("PUT", path, data, parsed.netloc, config)
    request = Request(url, data=data, headers=headers, method="PUT")
    try:
        with urlopen(request, timeout=10) as response:
            if response.status not in (200, 201):
                raise RuntimeError(f"MinIO upload failed with status {response.status}")
    except HTTPError as exc:
        if exc.code == 409 and not object_name:
            return
        raise RuntimeError(f"MinIO upload failed for {bucket}/{object_name}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"MinIO upload failed for {bucket}/{object_name}: {exc}") from exc


def _signed_headers(
    method: str,
    canonical_uri: str,
    data: bytes,
    host: str,
    config: MinioConfig,
) -> dict[str, str]:
    now = datetime.now(UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(data).hexdigest()
    credential_scope = f"{date_stamp}/{config['region']}/s3/aws4_request"
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        [method, canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _signature_key(config["secret_key"], date_stamp, config["region"], "s3")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={config['access_key']}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }


def _signature_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    key = ("AWS4" + secret_key).encode("utf-8")
    date_key = hmac.new(key, date_stamp.encode("utf-8"), hashlib.sha256).digest()
    region_key = hmac.new(date_key, region.encode("utf-8"), hashlib.sha256).digest()
    service_key = hmac.new(region_key, service.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(service_key, b"aws4_request", hashlib.sha256).digest()
