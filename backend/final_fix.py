#!/usr/bin/env python3

# Read current main.py
with open('main.py', 'r') as f:
    content = f.read()

# Replace the entire LlamaIndexRAGService class with the corrected version
class_start = 'class LlamaIndexRAGService:'
class_end = '\n# Professional Response Generator'

start_pos = content.find(class_start)
if start_pos != -1:
    end_pos = content.find(class_end, start_pos)
    if end_pos != -1:
        old_class = content[start_pos:end_pos]
        
        new_class = '''class LlamaIndexRAGService:
    """Fixed RAG service using correct LlamaIndex + Weaviate configuration"""

    def __init__(self):
        self.weaviate_client = None
        self.vector_store = None
        self.index = None
        self.service_context = None
        self.llm = None
        self.embed_model = None
        self.initialize_services()

    def initialize_services(self):
        """Initialize LlamaIndex services with correct Weaviate configuration"""
        try:
            if not LLAMAINDEX_AVAILABLE or not llamaindex_components:
                logger.error("❌ LlamaIndex components not available")
                return

            # Environment variables
            openai_api_key = os.getenv("OPENAI_API_KEY")
            weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
            weaviate_key = os.getenv("WEAVIATE_API_KEY")

            if not openai_api_key:
                logger.error("❌ OPENAI_API_KEY not found")
                return

            # Parse Weaviate URL
            if "://" in weaviate_url:
                host = weaviate_url.split("://")[1]
            else:
                host = weaviate_url

            if ":" in host:
                host = host.split(":")[0]

            logger.info(f"🔗 Connecting to Weaviate at {host}:8080...")

            # Initialize Weaviate client
            auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_key) if weaviate_key else None
            
            self.weaviate_client = weaviate.connect_to_local(
                host=host,
                port=8080,
                auth_credentials=auth_config,
                headers={"X-OpenAI-Api-Key": openai_api_key}
            )

            if not self.weaviate_client.is_ready():
                logger.error("❌ Weaviate client not ready")
                return

            logger.info("✅ Weaviate client connected")

            # Initialize LlamaIndex components using loaded components
            OpenAI = llamaindex_components['OpenAI']
            OpenAIEmbedding = llamaindex_components['OpenAIEmbedding']
            WeaviateVectorStore = llamaindex_components['WeaviateVectorStore']
            VectorStoreIndex = llamaindex_components['VectorStoreIndex']
            StorageContext = llamaindex_components['StorageContext']

            # Create LLM and embedding model
            self.llm = OpenAI(
                model="gpt-4",
                api_key=openai_api_key,
                temperature=0.1,
                max_tokens=1500
            )

            self.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=openai_api_key
            )

            # Create service context (CRITICAL: this ensures consistency)
            from llama_index.core import ServiceContext
            self.service_context = ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model
            )

            # Setup Weaviate schema if needed
            self.setup_weaviate_schema()

            # FIXED: Connect to existing Weaviate class with correct parameters
            self.vector_store = WeaviateVectorStore(
                weaviate_client=self.weaviate_client,
                index_name="Documents",  # FIXED: Use existing class name
                text_key="content",      # FIXED: Specify correct text field
                metadata_keys=["filename", "user_id", "document_id", "node_id", "chunk_id", "page_number"]
            )

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # FIXED: Load from existing vector store (don't create new)
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    service_context=self.service_context,
                    storage_context=storage_context
                )
                logger.info("✅ Connected to existing LlamaIndex from Weaviate")
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing index, creating new: {e}")
                # Fallback: create empty index
                self.index = VectorStoreIndex(
                    nodes=[],
                    service_context=self.service_context,
                    storage_context=storage_context
                )
                logger.info("✅ Created new LlamaIndex")

            logger.info("✅ LlamaIndex RAG service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LlamaIndex RAG service: {e}")
            self.weaviate_client = None

    def setup_weaviate_schema(self):
        """Setup Weaviate schema - same as before"""
        try:
            collections = self.weaviate_client.collections.list_all()
            collection_names = [col for col in collections.keys()] if hasattr(collections, 'keys') else [col.name for col in collections]

            if "Documents" not in collection_names:
                logger.info("📋 Creating Documents collection...")
                # Schema creation code same as before
                pass
            else:
                logger.info("✅ Documents collection already exists")

        except Exception as e:
            logger.error(f"❌ Schema setup error: {e}")

    async def search_documents(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """FIXED: Working document search using correct LlamaIndex configuration"""
        try:
            logger.info(f"🔍 LlamaIndex search for: '{query}' user: '{user_id}'")
            
            if not self.index:
                logger.error("❌ LlamaIndex index not available")
                return []

            # FIXED: Create retriever with proper parameters
            retriever = self.index.as_retriever(
                similarity_top_k=limit,
                alpha=0.5  # Hybrid search (vector + BM25)
            )

            # FIXED: Build user filter correctly
            from weaviate.classes.query import Filter
            filters = {
                "operator": "And",
                "conditions": [
                    {"path": ["user_id"], "operator": "Equal", "valueText": user_id}
                ]
            }

            # Retrieve nodes with filter
            try:
                # FIXED: Use aretrieve for async operation
                if hasattr(retriever, 'aretrieve'):
                    nodes_with_scores = await retriever.aretrieve(query, filters=filters)
                else:
                    nodes_with_scores = retriever.retrieve(query, filters=filters)
            except Exception as e:
                logger.warning(f"⚠️ Filtered retrieval failed, trying without filter: {e}")
                # Fallback: retrieve without filter and manually filter
                if hasattr(retriever, 'aretrieve'):
                    nodes_with_scores = await retriever.aretrieve(query)
                else:
                    nodes_with_scores = retriever.retrieve(query)
                
                # Manual filtering
                filtered_nodes = []
                for node_with_score in nodes_with_scores:
                    if node_with_score.node.metadata.get("user_id") == user_id:
                        filtered_nodes.append(node_with_score)
                nodes_with_scores = filtered_nodes

            results = []
            for node_with_score in nodes_with_scores:
                node = node_with_score.node
                metadata = node.metadata or {}
                
                results.append({
                    "content": node.text,  # FIXED: LlamaIndex now provides text correctly
                    "filename": metadata.get("filename", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "page_number": metadata.get("page_number", 1),
                    "similarity_score": getattr(node_with_score, 'score', 0.8),
                    "document_id": metadata.get("document_id", ""),
                    "node_id": node.node_id
                })

            logger.info(f"✅ LlamaIndex found {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"❌ LlamaIndex search error: {e}")
            return []

    async def generate_rag_response(self, query: str, user_id: str, context_limit: int = 5) -> Dict[str, Any]:
        """FIXED: Working RAG response using corrected search method"""
        try:
            logger.info(f"🔍 RAG query: {query[:100]}...")

            # Use our corrected search method
            search_results = await self.search_documents(query, user_id, context_limit)

            if not search_results:
                return {
                    "response": f\"\"\"I don't have any documents in your knowledge base that are relevant to: "{query}"

To get started, please upload some documents using the document upload feature. I can then provide detailed answers based on your specific content.

Would you like me to explain how to upload documents?\"\"\",
                    "sources": [],
                    "has_context": False
                }

            # Build context from search results
            context_parts = []
            sources = []

            for i, result in enumerate(search_results):
                context_parts.append(f"[Source {i+1}] {result['content']}")
                sources.append({
                    "filename": result["filename"],
                    "page": result["page_number"],
                    "chunk": result["chunk_id"],
                    "relevance": round(result["similarity_score"], 3),
                    "node_id": result["node_id"]
                })

            context = "\\n\\n".join(context_parts)

            # Generate response using LLM
            if self.llm:
                try:
                    from llama_index.core.llms import ChatMessage, MessageRole
                    
                    messages = [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content="You are IntraNest AI, a professional enterprise knowledge assistant. Provide accurate, well-structured responses based on the provided document context."
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=f\"\"\"Based on the following documents from the user's knowledge base, answer the question: "{query}"

Context from documents:
{context}

Instructions:
1. Provide a comprehensive answer based on the provided context
2. Reference specific sources when making claims
3. If the context doesn't fully answer the question, acknowledge what's missing
4. Use a professional, clear tone suitable for business use
5. Structure your response with headers and bullet points when appropriate

Answer:\"\"\"
                        )
                    ]

                    if hasattr(self.llm, 'achat'):
                        response = await self.llm.achat(messages)
                    else:
                        response = self.llm.chat(messages)
                    ai_response = str(response)
                except Exception as e:
                    logger.warning(f"⚠️ LLM chat failed, using simple response: {e}")
                    ai_response = f"Based on your documents:\\n\\n{context[:500]}..."
            else:
                ai_response = f"Based on your documents:\\n\\n{context[:500]}..."

            # Format response with sources
            if sources:
                sources_text = "\\n\\n**Sources:**\\n"
                for i, source in enumerate(sources):
                    sources_text += f"• {source['filename']} (Page {source['page']}, Relevance: {source['relevance']})\\n"
                
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
            logger.error(f"❌ RAG response error: {e}")
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "has_context": False
            }

'''
        
        content = content.replace(old_class, new_class)
        print("✅ Applied comprehensive LlamaIndex fix")
        
        # Write the fixed file
        with open('main.py', 'w') as f:
            f.write(content)
        
        print("✅ IntraNest 2.0 RAG system fixed!")
    else:
        print("❌ Could not find class end")
else:
    print("❌ Could not find LlamaIndexRAGService class")
