#!/usr/bin/env python3
"""
Services initialization for IntraNest 2.0
"""

import logging
from .storage_service import StorageService
from .cache_service import DocumentCacheService
from .document_processor import EnhancedDocumentProcessor
from .rag_service import LlamaIndexRAGService
from .response_service import ProfessionalResponseGenerator

logger = logging.getLogger(__name__)

# Global service instances
storage_service = None
cache_service = None
document_processor = None
rag_service = None
response_generator = None

async def initialize_all_services():
    """Initialize all services"""
    global storage_service, cache_service, document_processor, rag_service, response_generator
    
    try:
        logger.info("🚀 Initializing IntraNest services...")

        # Initialize storage service
        storage_service = StorageService()
        logger.info("✅ Storage service initialized")

        # Initialize cache service
        cache_service = DocumentCacheService()
        logger.info("✅ Cache service initialized")

        # Initialize document processor
        document_processor = EnhancedDocumentProcessor()
        logger.info("✅ Document processor initialized")

        # Initialize RAG service
        rag_service = LlamaIndexRAGService()
        logger.info("✅ RAG service initialized")

        # Initialize response generator
        response_generator = ProfessionalResponseGenerator()
        logger.info("✅ Response generator initialized")

        logger.info("✅ All services initialized successfully")

    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise

async def cleanup_services():
    """Cleanup all services"""
    global rag_service
    
    logger.info("🛑 Shutting down IntraNest services...")
    
    if rag_service:
        rag_service.close()
        
    logger.info("✅ Services cleanup complete")

def get_storage_service():
    return storage_service

def get_cache_service():
    return cache_service

def get_document_processor():
    return document_processor

def get_rag_service():
    return rag_service

def get_response_generator():
    return response_generator
