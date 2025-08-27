#!/usr/bin/env python3
"""
IntraNest 2.0 - Enterprise AI Knowledge Management
Modular FastAPI application with enhanced document processing and conversational RAG capabilities

Version: 2.0.2-conversational
Status: Production Ready with Modular Architecture + Conversational RAG
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from datetime import datetime

# Import configuration and core modules
from config.settings import get_settings
from core.middleware import setup_middleware

# Import services
from services import initialize_all_services, cleanup_services

# Import API routers
from api import documents, chat, debug
from api.conversational_chat import router as conversational_router  # NEW: Conversational RAG

# Import conversational dependencies for health checks
from core.conversational_dependencies import check_conversational_services  # NEW

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with service initialization"""
    logger.info("🚀 Starting IntraNest 2.0 with modular architecture + conversational RAG...")

    try:
        # Initialize all services
        await initialize_all_services()
        logger.info("✅ IntraNest 2.0 started successfully with all services")

        # NEW: Check conversational services
        try:
            conv_health = await check_conversational_services()
            logger.info(f"🧠 Conversational RAG services: {conv_health['overall_status']}")
            if conv_health['overall_status'] == 'healthy':
                logger.info("✅ Conversational RAG fully operational")
            else:
                logger.warning(f"⚠️ Conversational RAG degraded: {conv_health.get('unhealthy_services', [])}")
        except Exception as e:
            logger.warning(f"⚠️ Conversational RAG health check failed: {e}")

        yield

    except Exception as e:
        logger.error(f"❌ Failed to start IntraNest 2.0: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down IntraNest 2.0...")
        await cleanup_services()
        logger.info("✅ Shutdown complete")

# Create FastAPI application with lifespan
app = FastAPI(
    title=settings.app_name,
    description="Modular document processing and conversational RAG system with enhanced PDF processing and natural conversation flow",
    version="2.0.2-conversational",  # Updated version
    lifespan=lifespan
)

# Setup middleware (CORS, GZip, etc.)
setup_middleware(app)

# Include API routers with proper prefixes and tags
app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["documents"]
)

app.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"]
)

# NEW: Include conversational RAG router
app.include_router(
    conversational_router,
    tags=["conversational-rag"]
)

app.include_router(
    debug.router,
    prefix="/api/debug",
    tags=["debug"]
)

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": settings.app_name,
        "version": "2.0.2-conversational",  # Updated version
        "status": "operational",
        "architecture": "modular + conversational RAG",  # Updated
        "timestamp": datetime.now().isoformat(),
        "features": {
            "document_upload": "enabled",
            "enhanced_pdf_processing": "enabled",
            "document_processing": "enabled",
            "document_listing": "enabled",
            "rag_search": "enabled",
            "real_time_progress": "enabled",
            "modular_architecture": "enabled",
            "conversational_rag": "enabled",  # NEW
            "context_retention": "enabled",  # NEW
            "coreference_resolution": "enabled",  # NEW
            "conversation_memory": "enabled",  # NEW
            "query_rewriting": "enabled"  # NEW
        },
        "endpoints": {
            "documents": "/api/documents/*",
            "chat": "/chat/*",
            "conversational_chat": "/api/chat/conversational",  # NEW
            "conversational_stream": "/api/chat/conversational/stream",  # NEW
            "chat_sessions": "/api/chat/sessions",  # NEW
            "debug": "/api/debug/*",
            "health": "/api/health",
            "conversational_health": "/api/health/conversational",  # NEW
            "docs": "/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    from services import (
        get_storage_service, get_cache_service, get_document_processor,
        get_rag_service, get_response_generator
    )

    # Check all services
    storage_service = get_storage_service()
    cache_service = get_cache_service()
    document_processor = get_document_processor()
    rag_service = get_rag_service()
    response_generator = get_response_generator()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.2-conversational",  # Updated version
        "architecture": "modular + conversational RAG",  # Updated
        "services": {
            "api": "running",
            "storage": "operational" if storage_service else "unavailable",
            "cache": "operational" if cache_service else "unavailable",
            "document_processor": "operational" if document_processor else "unavailable",
            "rag_service": "operational" if rag_service else "unavailable",
            "response_generator": "operational" if response_generator else "unavailable",
            "weaviate": "operational" if rag_service and rag_service.weaviate_client else "unavailable"
        },
        "capabilities": {
            "enhanced_pdf_processing": bool(document_processor),
            "real_time_progress": bool(cache_service),
            "document_management": bool(storage_service),
            "rag_search": bool(rag_service),
            "professional_responses": bool(response_generator),
            "streaming_chat": True,
            "openai_compatible": True,
            "conversational_rag": True,  # NEW
            "context_retention": True,  # NEW
            "multi_turn_conversations": True  # NEW
        },
        "environment": {
            "openai_configured": bool(settings.openai_api_key),
            "weaviate_configured": bool(settings.weaviate_api_key),
            "redis_configured": bool(settings.redis_url)
        }
    }

# NEW: Conversational RAG health check endpoint
@app.get("/api/health/conversational")
async def conversational_health():
    """Health check for conversational RAG services"""
    return await check_conversational_services()

# Run the application
if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting {settings.app_name} 2.0.2-conversational")
    logger.info(f"📊 Environment: {'production' if not settings.debug else 'development'}")
    logger.info(f"🔧 Configuration: Modular architecture with conversational RAG capabilities")
    logger.info(f"🧠 Features: Context retention, coreference resolution, conversation memory")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level=settings.log_level.lower(),
        access_log=True
    )
