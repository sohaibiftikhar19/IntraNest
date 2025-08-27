#!/usr/bin/env python3
"""
Cache service for IntraNest 2.0
"""

import json
import logging
import redis
from typing import Dict, List, Optional
from datetime import datetime
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class DocumentCacheService:
    """Handle document metadata caching with REAL progress tracking"""
    
    def __init__(self):
        self._memory_cache = {}
        self.document_metadata_cache = {}  # Cache for document listings
        self.use_redis = False

        try:
            self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using memory cache: {e}")
            self.redis_client = None

    def cache_processing_status(self, document_id: str, status: Dict, ttl: int = 3600):
        """Cache document processing status"""
        try:
            if self.use_redis and self.redis_client:
                cache_key = f"processing:document:{document_id}"
                self.redis_client.setex(cache_key, ttl, json.dumps(status))
            else:
                self._memory_cache[document_id] = status
            logger.debug(f"📦 Cached status for {document_id}: {status.get('status', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ Cache write failed: {e}")
            self._memory_cache[document_id] = status

    def get_processing_status(self, document_id: str) -> Optional[Dict]:
        """Get document processing status"""
        try:
            if self.use_redis and self.redis_client:
                cache_key = f"processing:document:{document_id}"
                cached = self.redis_client.get(cache_key)
                return json.loads(cached) if cached else None
            else:
                return self._memory_cache.get(document_id)
        except Exception as e:
            logger.warning(f"⚠️ Cache read failed: {e}")
            return self._memory_cache.get(document_id)

    def update_progress(self, document_id: str, status: str, progress: int, message: str, **kwargs):
        """Update processing progress in real-time"""
        try:
            current_status = self.get_processing_status(document_id) or {}
            current_status.update({
                "status": status,
                "progress": progress,
                "message": message,
                "updated_at": datetime.now().isoformat(),
                **kwargs
            })
            self.cache_processing_status(document_id, current_status)
            logger.info(f"📊 Progress [{document_id}]: {progress}% - {message}")
        except Exception as e:
            logger.error(f"❌ Failed to update progress: {e}")

    def cache_document_metadata(self, user_id: str, document_id: str, metadata: Dict):
        """Cache document metadata for faster listings"""
        try:
            if user_id not in self.document_metadata_cache:
                self.document_metadata_cache[user_id] = {}

            # Enhanced metadata with proper data types
            enhanced_metadata = {
                "id": document_id,
                "filename": metadata.get("filename", "Unknown"),
                "size": int(metadata.get("file_size", 0)),
                "uploadDate": metadata.get("upload_date", datetime.now().isoformat()),
                "status": metadata.get("status", "completed"),
                "userId": user_id,
                "documentId": document_id,
                "chunks": int(metadata.get("chunks_created", 0)),
                "wordCount": int(metadata.get("word_count", 0)),
                "fileType": metadata.get("mime_type", "text/plain"),
                "file_size": int(metadata.get("file_size", 0)),
                "upload_date": metadata.get("upload_date", datetime.now().isoformat()),
                "processing_status": metadata.get("status", "completed")
            }

            self.document_metadata_cache[user_id][document_id] = enhanced_metadata
            logger.info(f"📦 Cached metadata for {document_id}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to cache document metadata: {e}")

    def get_user_documents(self, user_id: str) -> List[Dict]:
        """Get cached documents for user"""
        try:
            user_docs = self.document_metadata_cache.get(user_id, {})
            return list(user_docs.values())
        except Exception as e:
            logger.warning(f"⚠️ Failed to get cached documents: {e}")
            return []
