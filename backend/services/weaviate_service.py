"""
Simplified Weaviate Service with compatible method signatures
"""
import weaviate
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class WeaviateService:
    def __init__(self, url: str = "http://localhost:8080", api_key: str = None):
        self.url = url
        self.api_key = api_key
        self.client = None
        
    async def initialize(self):
        """Initialize Weaviate client"""
        try:
            if self.api_key:
                auth_config = weaviate.auth.AuthApiKey(self.api_key)
            else:
                auth_config = None
                
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080,
                auth_credentials=auth_config
            )
            
            # Test connection
            meta = self.client.get_meta()
            logger.info(f"✅ Connected to Weaviate {meta.get('version')}")
            
            # Check if Documents collection exists
            try:
                collections = self.client.collections.list_all()
                collection_names = [col.name for col in collections]
                
                if "Documents" in collection_names:
                    logger.info("✅ Documents collection exists")
                else:
                    logger.warning("⚠️  Documents collection not found")
                    
            except Exception as e:
                logger.warning(f"⚠️  Could not check collections: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Weaviate: {e}")
            raise
    
    async def setup_tenant(self, tenant_id: str) -> bool:
        """Dummy setup_tenant method for compatibility"""
        logger.info(f"⚠️  setup_tenant called for {tenant_id} - ignoring (multi-tenancy disabled)")
        return True
    
    async def upload_document_chunks(self, tenant_id: str, document_data: Dict, chunks: List[Dict]) -> str:
        """Upload document chunks (compatible signature with tenant_id)"""
        try:
            documents = self.client.collections.get("Documents")
            
            uploaded_count = 0
            for chunk in chunks:
                chunk_object = {
                    "content": chunk["text"],
                    "filename": document_data["filename"],
                    "chunk_id": chunk["chunk_id"],
                    "document_id": document_data["document_id"],
                    "user_id": document_data["user_id"],
                    "metadata": chunk.get("metadata", {}),
                    "created_at": datetime.utcnow().isoformat()
                }
                
                # Insert object
                result = documents.data.insert(properties=chunk_object)
                if result:
                    uploaded_count += 1
            
            logger.info(f"✅ Uploaded {uploaded_count} chunks for document {document_data['document_id']}")
            return document_data["document_id"]
            
        except Exception as e:
            logger.error(f"❌ Failed to upload document chunks: {e}")
            raise
    
    async def search_documents(self, tenant_id: str, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """Search documents (compatible signature with tenant_id)"""
        try:
            documents = self.client.collections.get("Documents")
            
            # Simple search without tenant isolation for now
            response = documents.query.near_text(
                query=query,
                limit=limit,
                return_metadata=['score']
            )
            
            results = []
            for obj in response.objects:
                # Filter by user_id manually
                if obj.properties.get("user_id") == user_id:
                    results.append({
                        "id": str(obj.uuid),
                        "content": obj.properties["content"],
                        "filename": obj.properties["filename"],
                        "chunk_id": obj.properties["chunk_id"],
                        "document_id": obj.properties["document_id"],
                        "similarity_score": getattr(obj.metadata, 'score', 0.0) if hasattr(obj, 'metadata') else 0.0,
                        "metadata": obj.properties.get("metadata", {})
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []
    
    async def generate_rag_response(self, tenant_id: str, user_id: str, query: str, limit: int = 5) -> Dict:
        """Generate RAG response (compatible signature with tenant_id)"""
        try:
            documents = self.client.collections.get("Documents")
            
            # Use generative search
            response = documents.generate.near_text(
                query=query,
                limit=limit,
                grouped_task=f"""
                Based on the context documents, provide a comprehensive answer to: "{query}"
                
                Guidelines:
                1. Use only information from the provided context
                2. Cite specific document names when making claims
                3. If information is incomplete, clearly state this
                4. Provide accurate and helpful responses
                
                Question: {query}
                Answer:
                """
            )
            
            # Filter results by user_id
            user_sources = []
            for obj in response.objects:
                if obj.properties.get("user_id") == user_id:
                    user_sources.append({
                        "filename": obj.properties["filename"],
                        "chunk_id": obj.properties["chunk_id"],
                        "content_preview": obj.properties["content"][:200] + "...",
                        "similarity_score": getattr(obj.metadata, 'score', 0.0) if hasattr(obj, 'metadata') else 0.0
                    })
            
            return {
                "response": response.generated if hasattr(response, 'generated') else "No response generated",
                "sources": user_sources,
                "document_context_used": len(user_sources) > 0,
                "metadata": {
                    "query": query,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ RAG generation failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "sources": [],
                "document_context_used": False,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict:
        """Check Weaviate health"""
        try:
            if not self.client:
                return {"status": "unhealthy", "error": "Client not initialized"}
                
            meta = self.client.get_meta()
            return {
                "status": "healthy",
                "version": meta.get("version"),
                "modules": list(meta.get("modules", {}).keys())
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self):
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
