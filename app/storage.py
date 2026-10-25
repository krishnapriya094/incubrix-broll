"""Storage adapter - 'files in, files out' (Integration Standard point 7).

Two backends behind one interface, chosen by STORAGE_BACKEND:
  * local - writes under DATA_DIR/outputs and returns PUBLIC_BASE_URL/files/... links.
  * s3    - writes to any S3-compatible store (MinIO in docker compose).
"""
from __future__ import annotations

import shutil
from pathlib import Path


class LocalStorage:
    def __init__(self, root: Path, public_base_url: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")

    def _path(self, key: str) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_bytes(self, key: str, data: bytes) -> str:
        self._path(key).write_bytes(data)
        return self.url(key)

    def save_file(self, key: str, source_path: str | Path) -> str:
        shutil.copyfile(str(source_path), str(self._path(key)))
        return self.url(key)

    def local_path(self, key: str) -> Path:
        return self.root / key

    def url(self, key: str) -> str:
        return f"{self.public_base_url}/files/{key}"

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()


class S3Storage:
    def __init__(self, settings):
        import boto3  # imported lazily so local mode has no hard dependency

        self.bucket = settings.s3_bucket
        self.public_base_url = settings.public_base_url.rstrip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def save_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return self.url(key)

    def save_file(self, key: str, source_path: str | Path) -> str:
        self.client.upload_file(str(source_path), self.bucket, key)
        return self.url(key)

    def local_path(self, key: str) -> Path:  # pragma: no cover - not used in s3 mode
        raise NotImplementedError("local_path is only available with the local backend")

    def url(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=7 * 24 * 3600
        )

    def exists(self, key: str) -> bool:  # pragma: no cover
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


def get_storage(settings):
    if settings.storage_backend == "s3":
        return S3Storage(settings)
    return LocalStorage(settings.data_dir / "outputs", settings.public_base_url)
