#!/usr/bin/env python3
"""
Storage service for IntraNest 2.0
"""

import os
import logging
import uuid
from pathlib import Path
from config.document_config import DocumentConfig

logger = logging.getLogger(__name__)

class StorageService:
    """Handle file storage operations"""
    
    def __init__(self):
        try:
            os.makedirs(DocumentConfig.UPLOAD_DIR, exist_ok=True)
            logger.info(f"✅ Storage service initialized: {DocumentConfig.UPLOAD_DIR}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize storage service: {e}")
            raise

    def save_uploaded_file(self, file_content: bytes, filename: str, user_id: str) -> str:
        """Save uploaded file locally and return file path"""
        try:
            # Create user-specific directory
            user_dir = os.path.join(DocumentConfig.UPLOAD_DIR, user_id)
            os.makedirs(user_dir, exist_ok=True)

            # Generate unique filename
            file_extension = Path(filename).suffix
            unique_filename = f"{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(user_dir, unique_filename)

            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)

            logger.info(f"✅ File saved: {file_path} ({len(file_content)} bytes)")
            return file_path

        except Exception as e:
            logger.error(f"❌ Failed to save file: {e}")
            raise
