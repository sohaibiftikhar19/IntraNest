#!/usr/bin/env python3
"""
Enhanced RAG service for IntraNest 2.0 with superior response quality
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

from config.settings import get_settings
from utils.weaviate_helper import WeaviateHelper

logger = logging.getLogger(__name__)
settings = get_settings()

# LlamaIndex imports - test each component individually
LLAMAINDEX_AVAILABLE = True
llamaindex_components = {}

try:
    from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.llms import ChatMessage, MessageRole
    
    llamaindex_components.update({
        'VectorStoreIndex': VectorStoreIndex,
        'Document': Document,
        'Settings': Settings,
        'StorageContext': StorageContext,
        'SentenceSplitter': SentenceSplitter,
        'WeaviateVectorStore': WeaviateVectorStore,
        'OpenAI': OpenAI,
        'OpenAIEmbedding': OpenAIEmbedding,
        'ChatMessage': ChatMessage,
        'MessageRole': MessageRole
    })
    logger.info("✅ Core LlamaIndex components loaded")
except ImportError as e:
    logger.error(f"❌ Core LlamaIndex components failed: {e}")
    LLAMAINDEX_AVAILABLE = False

class LlamaIndexRAGService:
    """Enhanced RAG service with superior response quality matching your original system"""

    def __init__(self):
        self.weaviate_client = None
        self.vector_store = None
        self.index = None
        self.llm = None
        self.embed_model = None
        self.initialize_services()

    def initialize_services(self):
        """Initialize LlamaIndex services"""
        try:
            logger.info("🚀 Initializing LlamaIndex RAG service...")

            if not LLAMAINDEX_AVAILABLE or not llamaindex_components:
                logger.error("❌ LlamaIndex components not available")
                return

            if not settings.openai_api_key:
                logger.error("❌ OPENAI_API_KEY not found")
                return

            # Initialize Weaviate client
            try:
                self.weaviate_client = WeaviateHelper.get_client()
                if not self.weaviate_client.is_ready():
                    logger.error("❌ Weaviate client not ready")
                    return
                logger.info("✅ Weaviate client connected")
            except Exception as e:
                logger.error(f"❌ Weaviate connection failed: {e}")
                return

            # Initialize LlamaIndex components
            OpenAI = llamaindex_components['OpenAI']
            OpenAIEmbedding = llamaindex_components['OpenAIEmbedding']
            WeaviateVectorStore = llamaindex_components['WeaviateVectorStore']
            VectorStoreIndex = llamaindex_components['VectorStoreIndex']
            StorageContext = llamaindex_components['StorageContext']

            # Create LLM and embedding model with optimal settings
            self.llm = OpenAI(
                model="gpt-4",
                api_key=settings.openai_api_key,
                temperature=0.2,  # Slightly higher for more natural responses
                max_tokens=2500   # More tokens for comprehensive responses
            )

            self.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key
            )

            # Configure Settings
            from llama_index.core import Settings as LISettings
            LISettings.llm = self.llm
            LISettings.embed_model = self.embed_model

            # Setup Weaviate schema
            WeaviateHelper.setup_weaviate_schema(self.weaviate_client)

            # Connect to existing Weaviate vector store
            self.vector_store = WeaviateVectorStore(
                weaviate_client=self.weaviate_client,
                index_name="Documents",
                text_key="content"
            )

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Load or create index
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    storage_context=storage_context
                )
                logger.info("✅ Connected to existing LlamaIndex")
            except Exception as e:
                logger.warning(f"⚠️ Creating new index: {e}")
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context
                )

            logger.info("✅ LlamaIndex RAG service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LlamaIndex RAG service: {e}")
            self.weaviate_client = None

    async def search_documents(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search documents with hybrid approach and robust property access"""
        try:
            logger.info(f"🔍 Searching documents for query: '{query[:50]}...' user: {user_id}")

            if not self.weaviate_client:
                logger.error("❌ No Weaviate client available")
                return []

            documents_collection = self.weaviate_client.collections.get("Documents")
            response = documents_collection.query.near_text(
                query=query,
                limit=limit * 3  # Get more results to filter by user
            )

            logger.debug(f"📄 Weaviate returned {len(response.objects)} total results")

            filtered_results = []
            for obj in response.objects:
                try:
                    props = obj.properties if hasattr(obj, 'properties') else {}
                    obj_user_id = WeaviateHelper.safe_get_property(props, 'user_id', '')

                    if obj_user_id == user_id:
                        # Calculate similarity score
                        similarity_score = 0.8  # Default
                        if hasattr(obj, 'metadata') and obj.metadata:
                            certainty = getattr(obj.metadata, 'certainty', None)
                            distance = getattr(obj.metadata, 'distance', None)

                            if certainty is not None:
                                similarity_score = float(certainty)
                            elif distance is not None:
                                similarity_score = max(0.0, 1.0 - float(distance))

                        result = {
                            "content": WeaviateHelper.safe_get_property(props, 'content', ''),
                            "filename": WeaviateHelper.safe_get_property(props, 'filename', 'Unknown'),
                            "chunk_id": WeaviateHelper.safe_get_property(props, 'chunk_id', 0),
                            "page_number": WeaviateHelper.safe_get_property(props, 'page_number', 1),
                            "similarity_score": float(similarity_score),
                            "document_id": WeaviateHelper.safe_get_property(props, 'document_id', ''),
                            "node_id": str(obj.uuid)
                        }

                        filtered_results.append(result)
                        logger.debug(f"✅ Added result: {result['filename']} (score: {similarity_score:.3f})")

                        if len(filtered_results) >= limit:
                            break

                except Exception as e:
                    logger.warning(f"⚠️ Error processing search result: {e}")
                    continue

            logger.info(f"✅ Search found {len(filtered_results)} documents for user {user_id}")
            return filtered_results

        except Exception as e:
            logger.error(f"❌ Document search error: {e}")
            return []

    async def generate_rag_response(self, query: str, user_id: str, context_limit: int = 5) -> Dict[str, Any]:
        """Generate professional RAG response matching your original system quality"""
        try:
            logger.info(f"🔍 Professional RAG query: {query[:100]}...")

            search_results = await self.search_documents(query, user_id, context_limit)

            if not search_results:
                return {
                    "response": f"""I don't have any documents in your knowledge base that are relevant to: "{query}"

To get started, please upload some documents using the document upload feature. I can then provide detailed answers based on your specific content.""",
                    "sources": [],
                    "has_context": False
                }

            # Build context from search results
            context_parts = []
            sources = []
            seen_content = set()

            for i, result in enumerate(search_results):
                content = result['content']
                content_key = content[:100].lower().strip()
                if content_key not in seen_content:
                    context_parts.append(f"Source {i+1}: {content}")
                    seen_content.add(content_key)

                    similarity_score = result.get("similarity_score", 0.8)
                    try:
                        relevance = round(float(similarity_score), 3)
                    except (ValueError, TypeError):
                        relevance = 0.800

                    sources.append({
                        "filename": result["filename"],
                        "page": result["page_number"],
                        "chunk": result["chunk_id"],
                        "relevance": relevance,
                        "node_id": result["node_id"]
                    })

            context = "\n\n".join(context_parts)

            # Generate professional response using optimized prompt
            if self.llm:
                try:
                    ChatMessage = llamaindex_components['ChatMessage']
                    MessageRole = llamaindex_components['MessageRole']

                    # Professional system prompt for enterprise knowledge assistant
                    system_prompt = """You are IntraNest AI, a professional enterprise knowledge assistant. You provide comprehensive, well-structured responses based on document context.

RESPONSE REQUIREMENTS:
- Use clear markdown formatting with **bold headers** and bullet points
- Structure information logically with sections like "Key Features", "Technical Details", etc.  
- Write in professional, business-appropriate language
- Be comprehensive but concise
- Use bullet points for lists and features
- Bold important terms and section headers
- End with a summary if appropriate

Format your response with proper markdown structure."""

                    user_prompt = f"""Based on the following documents from the user's knowledge base, answer the question: "{query}"

Context from documents:
{context}

Provide a comprehensive, professionally formatted response with clear sections and bullet points."""

                    messages = [
                        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                        ChatMessage(role=MessageRole.USER, content=user_prompt)
                    ]

                    response = self.llm.chat(messages)
                    ai_response = str(response)
                    logger.info("✅ Generated professional LLM response")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Professional LLM response failed, using fallback: {e}")
                    ai_response = f"Based on your documents:\n\n{context[:1000]}..."
            else:
                ai_response = f"Based on your documents:\n\n{context[:1000]}..."

            # Format final response with sources
            if sources:
                sources_text = "\n\n**Sources:**\n"
                for source in sources:
                    sources_text += f"• {source['filename']} (Page {source['page']}, Relevance: {source['relevance']})\n"
                final_response = ai_response + sources_text
            else:
                final_response = ai_response

            return {
                "response": final_response,
                "sources": sources,
                "has_context": True,
                "context_chunks": len(sources)
            }

        except Exception as e:
            logger.error(f"❌ Professional RAG response error: {e}")
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "has_context": False
            }

    def close(self):
        """Close Weaviate client"""
        if self.weaviate_client:
            try:
                self.weaviate_client.close()
            except:
                pass
