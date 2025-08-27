# services/conversational_rag.py
import asyncio
from typing import List, Dict, Optional, Any, Tuple
import weaviate
from services.memory_manager import MemoryManager
from services.query_rewriter import QueryRewriter
from models.conversation_models import (
    ChatMessage, ConversationState, QueryRewriteRequest, MessageRole, ConversationIntent
)
import logging
import openai
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationalRAGService:
    """Enhanced RAG service with conversation memory and context awareness"""

    def __init__(self,
                 memory_manager: MemoryManager,
                 query_rewriter: QueryRewriter,
                 weaviate_client: weaviate.Client,
                 openai_client: openai.AsyncOpenAI,
                 config: Dict[str, Any]):
        self.memory = memory_manager
        self.query_rewriter = query_rewriter
        self.weaviate = weaviate_client
        self.openai = openai_client
        self.config = config

        # Configuration
        self.max_retrieved_docs = config.get("max_retrieved_docs", 5)
        self.hybrid_alpha = config.get("hybrid_alpha", 0.7)  # Vector vs keyword balance
        self.response_model = config.get("response_model", "gpt-4")
        self.context_window_limit = config.get("context_window_limit", 4000)

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
            'salutations', 'bonjour', 'hola', 'aloha', 'morning', 'evening'
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
            'definitely', 'certainly', 'of course', 'cool', 'great'
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

    def _get_contextual_greeting(self) -> str:
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

    async def process_conversational_query(self,
                                         user_id: str,
                                         session_id: str,
                                         query: str,
                                         stream: bool = False) -> Dict[str, Any]:
        """Main method to process a query with full conversational context"""
        try:
            logger.info(f"Processing conversational query: {query[:50]}... for user {user_id}")
            
            # EARLY INTENT DETECTION - Check query intent before any processing
            query_intent = self._classify_query_intent(query)
            logger.info(f"Query intent classified as: {query_intent}")
            
            # Handle non-information queries WITHOUT retrieval or rewriting
            if query_intent == "greeting":
                greeting_response = self._get_contextual_greeting()
                
                # Add greeting to memory
                user_message = ChatMessage(
                    session_id=session_id,
                    content=query,
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                await self.memory.add_message(session_id, user_message)
                
                assistant_message = ChatMessage(
                    session_id=session_id,
                    content=greeting_response,
                    role=MessageRole.ASSISTANT,
                    timestamp=datetime.utcnow(),
                    metadata={"interaction_type": "greeting"}
                )
                await self.memory.add_message(session_id, assistant_message)
                
                return {
                    "type": "complete",
                    "response": greeting_response,
                    "metadata": {
                        "original_query": query,
                        "interaction_type": "greeting",
                        "skip_retrieval": True,
                        "skip_rewrite": True
                    }
                }
            
            elif query_intent == "farewell":
                farewell_response = "Goodbye! Feel free to return anytime if you have questions about your documents. Have a great day!"
                
                # Add to memory
                user_message = ChatMessage(
                    session_id=session_id,
                    content=query,
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                await self.memory.add_message(session_id, user_message)
                
                assistant_message = ChatMessage(
                    session_id=session_id,
                    content=farewell_response,
                    role=MessageRole.ASSISTANT,
                    timestamp=datetime.utcnow(),
                    metadata={"interaction_type": "farewell"}
                )
                await self.memory.add_message(session_id, assistant_message)
                
                return {
                    "type": "complete",
                    "response": farewell_response,
                    "metadata": {
                        "original_query": query,
                        "interaction_type": "farewell",
                        "skip_retrieval": True,
                        "skip_rewrite": True
                    }
                }
            
            elif query_intent == "acknowledgment":
                ack_response = "Is there anything else you'd like to know? I can help you search through your documents or answer questions about them."
                
                # Add to memory
                user_message = ChatMessage(
                    session_id=session_id,
                    content=query,
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                await self.memory.add_message(session_id, user_message)
                
                assistant_message = ChatMessage(
                    session_id=session_id,
                    content=ack_response,
                    role=MessageRole.ASSISTANT,
                    timestamp=datetime.utcnow(),
                    metadata={"interaction_type": "acknowledgment"}
                )
                await self.memory.add_message(session_id, assistant_message)
                
                return {
                    "type": "complete",
                    "response": ack_response,
                    "metadata": {
                        "original_query": query,
                        "interaction_type": "acknowledgment",
                        "skip_retrieval": True,
                        "skip_rewrite": True
                    }
                }
            
            elif query_intent == "help":
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
                
                # Add to memory
                user_message = ChatMessage(
                    session_id=session_id,
                    content=query,
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                await self.memory.add_message(session_id, user_message)
                
                assistant_message = ChatMessage(
                    session_id=session_id,
                    content=help_response,
                    role=MessageRole.ASSISTANT,
                    timestamp=datetime.utcnow(),
                    metadata={"interaction_type": "help"}
                )
                await self.memory.add_message(session_id, assistant_message)
                
                return {
                    "type": "complete",
                    "response": help_response,
                    "metadata": {
                        "original_query": query,
                        "interaction_type": "help",
                        "skip_retrieval": True,
                        "skip_rewrite": True
                    }
                }
            
            elif query_intent == "unclear":
                unclear_response = "I'm not quite sure what you're looking for. Could you please provide more details about what information you need from your documents?"
                
                # Add to memory
                user_message = ChatMessage(
                    session_id=session_id,
                    content=query,
                    role=MessageRole.USER,
                    timestamp=datetime.utcnow()
                )
                await self.memory.add_message(session_id, user_message)
                
                assistant_message = ChatMessage(
                    session_id=session_id,
                    content=unclear_response,
                    role=MessageRole.ASSISTANT,
                    timestamp=datetime.utcnow(),
                    metadata={"interaction_type": "unclear"}
                )
                await self.memory.add_message(session_id, assistant_message)
                
                return {
                    "type": "complete",
                    "response": unclear_response,
                    "metadata": {
                        "original_query": query,
                        "interaction_type": "unclear",
                        "skip_retrieval": True,
                        "skip_rewrite": True
                    }
                }

            # INFORMATION QUERY - Continue with full RAG processing
            # 1. Get conversation context from memory
            context = await self.memory.get_conversation_context(session_id, user_id)
            logger.info(f"Retrieved context with {len(context.get('recent_messages', []))} recent messages")

            # 2. Create user message and add to memory
            user_message = ChatMessage(
                session_id=session_id,
                content=query,
                role=MessageRole.USER,
                timestamp=datetime.utcnow()
            )
            await self.memory.add_message(session_id, user_message)
            logger.info("Added user message to memory")

            # 3. Rewrite query with context (only for information queries)
            conversation_state = context.get("conversation_state")
            if conversation_state is None:
                # Create a default conversation state if none exists
                conversation_state = ConversationState(session_id=session_id)
                logger.info("Created default conversation state")

            rewrite_request = QueryRewriteRequest(
                original_query=query,
                conversation_context=self._convert_context_to_messages(context["recent_messages"]),
                current_state=conversation_state
            )

            rewrite_response = await self.query_rewriter.rewrite_query_with_context(rewrite_request)
            logger.info(f"Query rewritten: '{query}' -> '{rewrite_response.rewritten_query}'")

            # 4. Perform hybrid retrieval
            retrieval_results = await self._hybrid_retrieval(
                user_id=user_id,
                original_query=query,
                rewritten_query=rewrite_response.rewritten_query,
                conversation_state=conversation_state
            )
            logger.info(f"Retrieved {len(retrieval_results['documents'])} documents")

            # 5. Build conversational prompt
            prompt_data = await self._build_conversational_prompt(
                query=query,
                rewritten_query=rewrite_response.rewritten_query,
                context=context,
                retrieved_docs=retrieval_results["documents"],
                resolved_entities=rewrite_response.resolved_entities
            )

            # 6. Generate response
            if stream:
                response_generator = self._generate_streaming_response(prompt_data)
                return {
                    "type": "stream",
                    "generator": response_generator,
                    "metadata": {
                        "original_query": query,
                        "rewritten_query": rewrite_response.rewritten_query,
                        "retrieved_docs": len(retrieval_results["documents"]),
                        "resolved_entities": rewrite_response.resolved_entities,
                        "confidence": rewrite_response.confidence_score,
                        "interaction_type": "information_query"
                    }
                }
            else:
                response_content = await self._generate_response(prompt_data)
                logger.info("Generated response successfully")

                # 7. Create assistant message and add to memory
                assistant_message = ChatMessage(
                    session_id=session_id,
                    content=response_content,
                    role=MessageRole.ASSISTANT,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "retrieved_docs": retrieval_results["document_ids"],
                        "rewritten_query": rewrite_response.rewritten_query,
                        "resolved_entities": rewrite_response.resolved_entities,
                        "interaction_type": "information_query"
                    }
                )
                await self.memory.add_message(session_id, assistant_message)
                logger.info("Added assistant response to memory")

                return {
                    "type": "complete",
                    "response": response_content,
                    "metadata": {
                        "original_query": query,
                        "rewritten_query": rewrite_response.rewritten_query,
                        "retrieved_docs": len(retrieval_results["documents"]),
                        "resolved_entities": rewrite_response.resolved_entities,
                        "confidence": rewrite_response.confidence_score,
                        "sources": retrieval_results["sources"],
                        "interaction_type": "information_query"
                    }
                }

        except Exception as e:
            logger.error(f"Error in conversational query processing: {e}")
            return {
                "type": "error",
                "error": str(e),
                "fallback_response": "I apologize, but I encountered an error processing your question. Could you please rephrase it?"
            }

    async def _hybrid_retrieval(self,
                              user_id: str,
                              original_query: str,
                              rewritten_query: str,
                              conversation_state: Optional[ConversationState]) -> Dict[str, Any]:
        """Perform hybrid retrieval combining multiple strategies"""
        try:
            # Use rewritten query as primary search
            primary_query = rewritten_query if rewritten_query != original_query else original_query

            # For now, we'll use a simplified retrieval since Weaviate client is None
            # In the future, this will implement full hybrid search

            # Fallback to using existing RAG service through HTTP requests
            logger.info(f"Using fallback retrieval for query: {primary_query}")

            # Simple document retrieval (would be enhanced with proper Weaviate integration)
            documents = await self._simple_document_retrieval(user_id, primary_query)

            return {
                "documents": documents,
                "document_ids": [doc.get("id", "") for doc in documents],
                "sources": [doc.get("source", "Unknown") for doc in documents],
                "total_results": len(documents),
                "search_strategy": "fallback_simple"
            }

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return {"documents": [], "document_ids": [], "sources": [], "total_results": 0}

    async def _simple_document_retrieval(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """Simple document retrieval as fallback"""
        try:
            # Check if this is still a greeting that somehow got through
            if query.lower().strip() in ['hello', 'hi', 'hey']:
                return []  # Return empty documents for greetings
            
            # This is a simplified version - in production you'd use proper Weaviate integration
            # For now, return mock documents to test the conversation flow
            # Only return TCS documents if the query actually mentions TCS or related terms
            tcs_keywords = ['tcs', 'transit', 'cyber', 'solutions', 'cybersecurity', 'training']
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in tcs_keywords):
                return [
                    {
                        "id": "doc1",
                        "content": "Transit Cyber Solutions (TCS) is a cybersecurity training company that specializes in public transportation.",
                        "source": "TCS Overview",
                        "metadata": {"relevance": 0.9},
                        "score": 0.9
                    },
                    {
                        "id": "doc2",
                        "content": "TCS provides AI-powered training platforms with customizable scenarios for transit employees.",
                        "source": "TCS Features",
                        "metadata": {"relevance": 0.8},
                        "score": 0.8
                    }
                ]
            else:
                # Return empty for queries that don't match any documents
                return []
                
        except Exception as e:
            logger.error(f"Simple document retrieval error: {e}")
            return []

    async def _build_conversational_prompt(self,
                                         query: str,
                                         rewritten_query: str,
                                         context: Dict[str, Any],
                                         retrieved_docs: List[Dict[str, Any]],
                                         resolved_entities: Dict[str, str]) -> Dict[str, str]:
        """Build a conversational prompt incorporating all context"""

        # System prompt
        system_prompt = """You are IntraNest AI, an intelligent assistant that helps users find and understand information from their organization's documents. You maintain conversation context and provide natural, helpful responses.

Key Behaviors:
1. Use conversation context naturally - reference previous discussions when relevant
2. When users ask follow-up questions, understand they're continuing the same topic
3. Provide specific, actionable answers based on the retrieved documents
4. If you resolved pronouns or references, acknowledge the context naturally
5. Cite sources clearly and encourage users to access full documents when helpful
6. Maintain a professional but conversational tone

You have access to the user's conversation history and can reference previous topics naturally."""

        # Build context section
        context_parts = []

        # Conversation context
        if context.get("context_summary"):
            context_parts.append(f"CONVERSATION CONTEXT:\n{context['context_summary']}")

        # Recent messages for immediate context
        if context.get("recent_messages"):
            recent_msgs = context["recent_messages"][-3:]  # Last 3 messages
            msg_context = []
            for msg in recent_msgs:
                role = msg.get("role", "").upper()
                content = msg.get("content", "")[:150]  # Truncate
                msg_context.append(f"{role}: {content}")
            context_parts.append(f"RECENT CONVERSATION:\n" + "\n".join(msg_context))

        # Resolved entities
        if resolved_entities:
            entities_text = ", ".join([f"{k}→{v}" for k, v in resolved_entities.items()])
            context_parts.append(f"RESOLVED REFERENCES: {entities_text}")

        # Retrieved documents
        if retrieved_docs:
            doc_context = "RETRIEVED INFORMATION:\n"
            for i, doc in enumerate(retrieved_docs[:5], 1):
                source = doc.get("source", "Unknown")
                content = doc.get("content", "")[:400]  # Limit content length
                doc_context += f"\n[Source {i}: {source}]\n{content}\n"
            context_parts.append(doc_context)
        else:
            context_parts.append("RETRIEVED INFORMATION:\nNo relevant documents found for this query.")

        # User query section
        query_section = f"USER QUESTION: {query}"
        if rewritten_query != query:
            query_section += f"\n(Interpreted as: {rewritten_query})"

        # Combine all parts
        user_prompt = "\n\n".join(context_parts + [query_section])

        # Instructions
        instructions = """
INSTRUCTIONS:
- Answer the user's question using the retrieved information and conversation context
- If no relevant documents were found, acknowledge this and offer to help in other ways
- Reference previous parts of our conversation naturally when relevant
- Be specific and cite sources clearly when available
- If the question builds on previous discussion, acknowledge that connection
- Provide actionable information when possible
- Keep response focused and conversational"""

        user_prompt += instructions

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    async def _generate_response(self, prompt_data: Dict[str, str]) -> str:
        """Generate response using OpenAI"""
        try:
            response = await self.openai.chat.completions.create(
                model=self.response_model,
                messages=[
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": prompt_data["user"]}
                ],
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try rephrasing your question."

    async def _generate_streaming_response(self, prompt_data: Dict[str, str]):
        """Generate streaming response"""
        try:
            stream = await self.openai.chat.completions.create(
                model=self.response_model,
                messages=[
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": prompt_data["user"]}
                ],
                temperature=0.7,
                max_tokens=800,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield "I apologize, but I encountered an error. Please try again."

    def _convert_context_to_messages(self, context_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
        """Convert context message dicts to ChatMessage objects"""
        messages = []
        for msg_data in context_messages:
            try:
                message = ChatMessage(
                    session_id=msg_data.get("session_id", ""),
                    content=msg_data.get("content", ""),
                    role=MessageRole(msg_data.get("role", "user")),
                    entities=msg_data.get("entities", {})
                )
                messages.append(message)
            except Exception as e:
                logger.warning(f"Could not convert message to ChatMessage: {e}")
                continue
        return messages
