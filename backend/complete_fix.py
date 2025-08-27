#!/usr/bin/env python3

# Read the current main.py
with open('main.py', 'r') as f:
    content = f.read()

# Replace the entire generate_rag_response method
old_rag_method_start = '''    async def generate_rag_response(self, query: str, user_id: str, context_limit: int = 5) -> Dict[str, Any]:
        """Generate RAG response using LlamaIndex query engine"""'''

# Find the end of the method (next method definition)
start_pos = content.find(old_rag_method_start)
if start_pos != -1:
    # Find the next method definition
    next_method_pos = content.find('\n    async def ', start_pos + 1)
    if next_method_pos == -1:
        next_method_pos = content.find('\n    def ', start_pos + 1)
    
    if next_method_pos != -1:
        # Replace the entire method
        old_rag_method = content[start_pos:next_method_pos]
        
        new_rag_method = '''    async def generate_rag_response(self, query: str, user_id: str, context_limit: int = 5) -> Dict[str, Any]:
        """Generate RAG response using our working search method"""
        try:
            logger.info(f"🔍 RAG query: {query[:100]}...")

            # Use our working search method instead of problematic LlamaIndex query engine
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

            # Generate response using OpenAI with context (bypass problematic query engine)
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
            }'''

        content = content.replace(old_rag_method, new_rag_method)
        print("✅ Replaced generate_rag_response method")
    else:
        print("❌ Could not find end of generate_rag_response method")
else:
    print("❌ Could not find generate_rag_response method")

# Write the fixed file
with open('main.py', 'w') as f:
    f.write(content)

print("✅ Applied complete RAG fix")
