"""Storage backend service providing Local and S3 storage abstractions.

Per 03 §1.4:
  - LocalStorageBackend: files stored under configured root directory
  - S3StorageBackend: S3-compatible object storage via boto3
  - Abstracted behind StorageBackend interface so all call sites share one signature

IMPORTANT: This module belongs in app/services.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from typing import BinaryIO, Protocol

import structlog

from app.config import Settings, get_settings

log = structlog.get_logger()


class StorageBackend(Protocol):
    """Abstract interface for object storage."""

    def put(self, key: str, data: bytes | BinaryIO, content_type: str = "application/octet-stream") -> str:
        """Store data under the specified key. Returns the storage key."""
        ...

    def get(self, key: str) -> bytes:
        """Retrieve full byte content for a key. Raises FileNotFoundError if missing."""
        ...

    def delete(self, key: str) -> bool:
        """Delete an object by key. Returns True if deleted or already absent."""
        ...

    def exists(self, key: str) -> bool:
        """Check if an object exists."""
        ...

    def get_url(self, key: str, expires_in_seconds: int = 3600) -> str:
        """Generate a direct or signed URL to access the artifact."""
        ...


class LocalStorageBackend:
    """Local filesystem storage implementation."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Strip leading slashes to prevent root escaping
        clean_key = key.lstrip("/\\")
        path = (self.root / clean_key).resolve()
        if not str(path).startswith(str(self.root)):
            raise ValueError(f"Directory traversal detected in storage key: {key}")
        return path

    def put(self, key: str, data: bytes | BinaryIO, content_type: str = "application/octet-stream") -> str:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, bytes):
            dest.write_bytes(data)
        else:
            with open(dest, "wb") as f:
                shutil.copyfileobj(data, f)

        log.debug("storage_local_put", key=key, path=str(dest))
        return key

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(f"Storage object not found: {key}")
        return path.read_bytes()

    def delete(self, key: str) -> bool:
        try:
            path = self._resolve(key)
            if path.is_file():
                path.unlink()
            return True
        except Exception as e:
            log.warning("storage_local_delete_failed", key=key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except Exception:
            return False

    def get_url(self, key: str, expires_in_seconds: int = 3600) -> str:
        # Returns internal API artifact URL
        return f"/api/v1/artifacts/{key}"


class S3StorageBackend:
    """S3-compatible object storage implementation using boto3."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region_name: str = "us-east-1",
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def put(self, key: str, data: bytes | BinaryIO, content_type: str = "application/octet-stream") -> str:
        s3 = self._get_client()
        body = data if isinstance(data, bytes) else data.read()
        s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)
        return key

    def get(self, key: str) -> bytes:
        s3 = self._get_client()
        response = s3.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def delete(self, key: str) -> bool:
        s3 = self._get_client()
        s3.delete_object(Bucket=self.bucket, Key=key)
        return True

    def exists(self, key: str) -> bool:
        s3 = self._get_client()
        try:
            s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def get_url(self, key: str, expires_in_seconds: int = 3600) -> str:
        s3 = self._get_client()
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in_seconds,
        )


_storage_instance: StorageBackend | None = None


def get_storage(settings: Settings | None = None) -> StorageBackend:
    """Singleton getter for the configured storage backend."""
    global _storage_instance
    if _storage_instance is None:
        cfg = settings or get_settings()
        if cfg.storage_backend == "s3" and cfg.s3_bucket:
            _storage_instance = S3StorageBackend(
                bucket=cfg.s3_bucket,
                endpoint_url=cfg.s3_endpoint,
                access_key=cfg.s3_access_key,
                secret_key=cfg.s3_secret_key,
            )
        else:
            _storage_instance = LocalStorageBackend(root_dir=cfg.storage_local_root)
    return _storage_instance
