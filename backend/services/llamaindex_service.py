"""
LlamaIndex Service for IntraNest 2.0
Provides RAG functionality using LlamaIndex components
Enhanced with query intent classification to handle greetings and simple interactions
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import os

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

import weaviate
from weaviate.auth import AuthApiKey

logger = logging.getLogger(__name__)

class LlamaIndexService:
    """LlamaIndex service for IntraNest 2.0 with enhanced query handling"""

    def __init__(self,
                 weaviate_url: str = "http://localhost:8080",
                 weaviate_api_key: str = None,
                 openai_api_key: str = None):

        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.openai_api_key = openai_api_key

        # Initialize components
        self.weaviate_client = self._init_weaviate_client()
        self._configure_settings()

        # Initialize vector store
        self.vector_store = WeaviateVectorStore(
            weaviate_client=self.weaviate_client,
            index_name="Documents"
        )

        # Document processor
        self.text_splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Storage for indices by user
        self.user_indices = {}

    def _init_weaviate_client(self):
        """Initialize Weaviate client"""
        try:
            auth_config = None
            if self.weaviate_api_key:
                auth_config = AuthApiKey(api_key=self.weaviate_api_key)

            # Parse the URL correctly
            weaviate_host = self.weaviate_url.replace("http://", "").replace("https://", "")
            if ":" in weaviate_host:
                host, port = weaviate_host.split(":", 1)
                port = int(port)
            else:
                host = weaviate_host
                port = 8080

            client = weaviate.connect_to_local(
                host=host,
                port=port,
                auth_credentials=auth_config
            )

            if client.is_ready():
                logger.info("✅ Weaviate client connected successfully")
                return client
            else:
                raise Exception("Weaviate is not ready")

        except Exception as e:
            logger.error(f"❌ Failed to connect to Weaviate: {e}")
            raise

    def _configure_settings(self):
        """Configure global LlamaIndex settings"""
        try:
            Settings.llm = OpenAI(
                api_key=self.openai_api_key,
                model="gpt-4",
                temperature=0.1,
                timeout=30
            )

            Settings.embed_model = OpenAIEmbedding(
                api_key=self.openai_api_key,
                model="text-embedding-3-small",
                timeout=30
            )

            logger.info("✅ LlamaIndex settings configured")

        except Exception as e:
            logger.error(f"❌ Failed to configure LlamaIndex settings: {e}")
            raise

    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the intent of a query to determine appropriate handling
        Returns: 'greeting', 'farewell', 'acknowledgment', 'help', 'unclear', or 'information_query'
        """
        if not query:
            return "unclear"
            
        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)
        
        # Greeting patterns
        greetings = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 
            'good evening', 'greetings', 'howdy', 'yo', 'sup', 'hiya',
            'salutations', 'bonjour', 'hola', 'aloha'
        ]
        
        # Farewell patterns  
        farewells = [
            'bye', 'goodbye', 'see you', 'farewell', 'later', 
            'see ya', 'take care', 'goodnight', 'night', 'adios',
            'ciao', 'au revoir', 'catch you later'
        ]
        
        # Simple acknowledgments
        acknowledgments = [
            'yes', 'no', 'okay', 'ok', 'sure', 'thanks', 
            'thank you', 'got it', 'understood', 'alright',
            'yep', 'nope', 'yeah', 'nah', 'indeed', 'absolutely',
            'definitely', 'certainly', 'of course'
        ]
        
        # Help requests
        help_keywords = [
            'help', 'assist', 'support', 'what can you do',
            'how do i', 'how can i', 'how to', 'can you help',
            'what do you do', 'capabilities', 'features'
        ]
        
        # Check for greetings (typically short phrases)
        if word_count <= 3:
            if any(greeting in query_lower for greeting in greetings):
                return "greeting"
            if any(farewell in query_lower for farewell in farewells):
                return "farewell"
            if query_lower in acknowledgments:
                return "acknowledgment"
        
        # Check for help requests
        if any(help_word in query_lower for help_word in help_keywords):
            return "help"
        
        # Check if query is too vague or short
        if word_count < 3 and query_lower not in acknowledgments:
            # But allow specific short queries that might be document searches
            if any(char in query for char in ['?', 'what', 'where', 'when', 'who', 'why', 'how']):
                return "information_query"
            return "unclear"
        
        # Default to information query for longer, substantive queries
        return "information_query"

    def _get_contextual_greeting(self, time_of_day: Optional[str] = None) -> str:
        """Generate a contextual greeting response"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            greeting_prefix = "Good morning!"
        elif 12 <= current_hour < 17:
            greeting_prefix = "Good afternoon!"
        elif 17 <= current_hour < 22:
            greeting_prefix = "Good evening!"
        else:
            greeting_prefix = "Hello!"
        
        return f"{greeting_prefix} Welcome to IntraNest AI. How can I assist you with your documents today?"

    async def ingest_document(self,
                            text_content: str,
                            filename: str,
                            user_id: str,
                            metadata: Dict = None) -> Dict[str, Any]:
        """Ingest a document using LlamaIndex"""
        try:
            logger.info(f"🔄 Starting document ingestion: {filename} for user {user_id}")

            document_metadata = {
                "filename": filename,
                "user_id": user_id,
                "ingestion_timestamp": datetime.utcnow().isoformat(),
                "document_id": str(uuid.uuid4()),
                **(metadata or {})
            }

            document = Document(
                text=text_content,
                metadata=document_metadata
            )

            index = await self._get_user_index(user_id)
            index.insert(document)

            nodes = self.text_splitter.get_nodes_from_documents([document])

            result = {
                "status": "success",
                "document_id": document_metadata["document_id"],
                "filename": filename,
                "chunks_created": len(nodes),
                "character_count": len(text_content),
                "word_count": len(text_content.split()),
                "metadata": document_metadata
            }

            logger.info(f"✅ Document ingested successfully: {filename} ({len(nodes)} chunks)")
            return result

        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "filename": filename
            }

    async def _get_user_index(self, user_id: str) -> VectorStoreIndex:
        """Get or create VectorStoreIndex for user"""
        if user_id not in self.user_indices:
            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store
                )
                self.user_indices[user_id] = index
                logger.info(f"✅ Created new index for user: {user_id}")

            except Exception as e:
                logger.error(f"❌ Failed to create index for user {user_id}: {e}")
                raise

        return self.user_indices[user_id]

    async def query_documents(self,
                            query: str,
                            user_id: str,
                            max_chunks: int = 5) -> Dict[str, Any]:
        """Query documents using RAG with LlamaIndex - Enhanced with intent classification"""
        try:
            logger.info(f"🔍 Processing query for user {user_id}: {query}")
            
            # Classify the query intent
            intent = self._classify_query_intent(query)
            logger.info(f"📊 Query classified as: {intent}")
            
            # Handle different intents without document search
            if intent == "greeting":
                greeting_response = self._get_contextual_greeting()
                return {
                    "status": "success",
                    "response": greeting_response,
                    "query": query,
                    "sources": [],
                    "context_used": False,
                    "metadata": {
                        "chunks_retrieved": 0,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interaction_type": "greeting"
                    }
                }
            
            elif intent == "farewell":
                return {
                    "status": "success",
                    "response": "Goodbye! Feel free to return anytime if you have questions about your documents. Have a great day!",
                    "query": query,
                    "sources": [],
                    "context_used": False,
                    "metadata": {
                        "chunks_retrieved": 0,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interaction_type": "farewell"
                    }
                }
            
            elif intent == "acknowledgment":
                return {
                    "status": "success",
                    "response": "Is there anything else you'd like to know? I can help you search through your documents or answer questions about them.",
                    "query": query,
                    "sources": [],
                    "context_used": False,
                    "metadata": {
                        "chunks_retrieved": 0,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interaction_type": "acknowledgment"
                    }
                }
            
            elif intent == "help":
                help_response = """I'm IntraNest AI, your document assistant. Here's how I can help you:

📄 **Document Search**: Ask me questions about any documents you've uploaded
🔍 **Information Retrieval**: I'll search through your documents to find relevant information
💡 **Intelligent Answers**: I provide context-aware responses based on your document content
📚 **Multi-Document Analysis**: I can synthesize information from multiple documents

Simply ask me a question about your documents, and I'll search through them to provide you with accurate, relevant answers!

For example, you could ask:
- "What does the report say about Q3 revenue?"
- "Summarize the main points from the project proposal"
- "Find information about security protocols"

What would you like to know about your documents?"""
                
                return {
                    "status": "success",
                    "response": help_response,
                    "query": query,
                    "sources": [],
                    "context_used": False,
                    "metadata": {
                        "chunks_retrieved": 0,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interaction_type": "help"
                    }
                }
            
            elif intent == "unclear":
                return {
                    "status": "success",
                    "response": "I'm not quite sure what you're looking for. Could you please provide more details about what information you need from your documents? For example, you could ask about specific topics, summaries, or search for particular content.",
                    "query": query,
                    "sources": [],
                    "context_used": False,
                    "metadata": {
                        "chunks_retrieved": 0,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interaction_type": "unclear"
                    }
                }
            
            # For information queries, proceed with document search
            logger.info(f"📚 Performing document search for: {query}")
            
            index = await self._get_user_index(user_id)

            # Use the query engine directly from index
            query_engine = index.as_query_engine(
                similarity_top_k=max_chunks,
                response_mode="compact"
            )

            response = query_engine.query(query)

            # Extract sources with better error handling
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    try:
                        if hasattr(node, 'metadata') and node.metadata.get('user_id') == user_id:
                            # Ensure we have valid text content
                            node_text = getattr(node, 'text', '') or getattr(node, 'get_content', lambda: '')()
                            if node_text:
                                sources.append({
                                    "filename": node.metadata.get('filename', 'Unknown'),
                                    "document_id": node.metadata.get('document_id'),
                                    "score": getattr(node, 'score', 0),
                                    "content_preview": node_text[:200] + "..." if len(node_text) > 200 else node_text
                                })
                    except Exception as e:
                        logger.warning(f"Error processing source node: {e}")
                        continue

            # Check if we found relevant sources
            if not sources:
                fallback_response = (
                    f"I couldn't find specific information about '{query}' in your documents. "
                    "This might be because:\n"
                    "1. The information isn't in your uploaded documents\n"
                    "2. The query needs to be more specific\n"
                    "3. The documents haven't been indexed yet\n\n"
                    "Try rephrasing your question or uploading relevant documents."
                )
                
                result = {
                    "status": "success",
                    "response": fallback_response,
                    "query": query,
                    "sources": [],
                    "context_used": False,
                    "metadata": {
                        "chunks_retrieved": 0,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interaction_type": "no_results"
                    }
                }
            else:
                result = {
                    "status": "success",
                    "response": str(response),
                    "query": query,
                    "sources": sources,
                    "context_used": True,
                    "metadata": {
                        "chunks_retrieved": len(sources),
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interaction_type": "information_query"
                    }
                }

            logger.info(f"✅ Query processed successfully. Found {len(sources)} relevant sources")
            return result

        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or rephrase your query.",
                "sources": [],
                "context_used": False,
                "metadata": {
                    "chunks_retrieved": 0,
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "interaction_type": "error"
                }
            }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for LlamaIndex service"""
        try:
            weaviate_ready = self.weaviate_client.is_ready()

            # More robust OpenAI check
            openai_ready = False
            openai_error = None
            try:
                # Test embedding generation with a simple word
                embedding = Settings.embed_model.get_text_embedding("test")
                openai_ready = len(embedding) > 0
            except Exception as e:
                openai_error = str(e)
                logger.warning(f"OpenAI health check failed: {e}")

            status = "healthy" if (weaviate_ready and openai_ready) else "degraded"

            result = {
                "status": status,
                "components": {
                    "weaviate": "ready" if weaviate_ready else "not_ready",
                    "openai": "ready" if openai_ready else "not_ready",
                    "llamaindex": "ready"
                },
                "active_indices": len(self.user_indices),
                "timestamp": datetime.utcnow().isoformat(),
                "features": {
                    "query_intent_classification": True,
                    "contextual_greetings": True,
                    "help_system": True,
                    "document_search": True
                }
            }

            if openai_error:
                result["openai_error"] = openai_error

            return result

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def close(self):
        """Close connections properly"""
        try:
            if hasattr(self, 'weaviate_client'):
                self.weaviate_client.close()
                logger.info("✅ Weaviate client closed successfully")
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
            pass
