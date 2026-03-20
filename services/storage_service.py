import os
import logging
from typing import Optional
from google.cloud import storage

class StorageService:
    """Service for Google Cloud Storage interactions."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        self.client = None
        if self.bucket_name:
            try:
                self.client = storage.Client()
                logging.info(f"GCS Storage Service initialized for bucket: {self.bucket_name}")
            except Exception as e:
                logging.warning(f"GCS Storage initialization failed (fallback to local): {e}")

    def upload_file(self, file_data: bytes, filename: str, content_type: str) -> Optional[str]:
        """Uploads a file to GCS and returns the public URL or GCS URI."""
        if not self.client or not self.bucket_name:
            logging.info("GCS not configured, skipping upload.")
            return None
        
        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_string(file_data, content_type=content_type)
            logging.info(f"File uploaded to GCS: {filename}")
            return f"gs://{self.bucket_name}/{filename}"
        except Exception as e:
            logging.error(f"Failed to upload to GCS: {e}")
            return None
