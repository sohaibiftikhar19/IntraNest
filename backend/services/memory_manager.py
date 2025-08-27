# services/memory_manager.py
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
import weaviate
import openai
from models.conversation_models import (
    ChatSession, ChatMessage, ConversationState,
    ConversationMemory, MessageRole, ConversationIntent
)
from utils.conversational_text_processing import ConversationalTextProcessor
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Dual-layer conversation memory management"""

    def __init__(self,
                 redis_client: redis.Redis,
                 weaviate_client: weaviate.Client,
                 db_session: AsyncSession,
                 openai_client: openai.AsyncOpenAI,  # Add OpenAI client parameter
                 config: Dict[str, Any]):
        self.redis = redis_client
        self.weaviate = weaviate_client
        self.db = db_session
        self.config = config
        
        # Use the passed OpenAI client instead of creating a new one
        self.text_processor = ConversationalTextProcessor(
            openai_client,
            config
        )

        # Configuration
        self.short_term_limit = config.get("short_term_limit", 10)
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour
        self.summary_threshold = config.get("summary_threshold", 5)

    async def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add new message to both short-term and persistent storage"""
        try:
            # 1. Add to short-term cache
            await self._add_to_short_term(session_id, message)

            # 2. Add to database
            await self._persist_message(message)

            # 3. Update conversation state
            await self._update_conversation_state(session_id, message)

            # 4. Check if summarization is needed
            message_count = await self._get_message_count(session_id)
            if message_count > self.summary_threshold:
                await self._trigger_summarization(session_id)

        except Exception as e:
            logger.error(f"Error adding message to memory: {e}")
            raise

    async def get_conversation_context(self,
                                     session_id: str,
                                     user_id: str,
                                     max_messages: int = 10) -> Dict[str, Any]:
        """Get conversation context combining short-term and long-term memory"""
        try:
            # 1. Get recent messages from cache
            recent_messages = await self._get_from_short_term(session_id, max_messages)

            # 2. Get conversation state
            state = await self._get_conversation_state(session_id)

            # 3. Get relevant long-term memories
            long_term_context = await self._get_relevant_memories(user_id, state)

            # 4. Combine context
            context = {
                "session_id": session_id,
                "recent_messages": recent_messages,
                "conversation_state": state,
                "long_term_memories": long_term_context,
                "context_summary": await self._build_context_summary(
                    recent_messages, long_term_context, state
                )
            }

            return context

        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {"recent_messages": [], "conversation_state": None}

    async def _add_to_short_term(self, session_id: str, message: ChatMessage) -> None:
        """Add message to Redis short-term cache"""
        cache_key = f"chat:session:{session_id}"

        # Get current messages
        cached_data = await self.redis.get(cache_key)
        messages = json.loads(cached_data) if cached_data else []

        # Add new message
        messages.append({
            "id": message.id,
            "content": message.content,
            "role": message.role.value,
            "timestamp": message.timestamp.isoformat(),
            "intent": message.intent.value if message.intent else None,
            "entities": message.entities
        })

        # Maintain sliding window
        if len(messages) > self.short_term_limit:
            # Move oldest messages to summarization queue
            oldest_messages = messages[:-self.short_term_limit]
            await self._queue_for_summarization(session_id, oldest_messages)
            messages = messages[-self.short_term_limit:]

        # Update cache
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(messages)
        )

    async def _get_from_short_term(self,
                                  session_id: str,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from cache"""
        cache_key = f"chat:session:{session_id}"
        cached_data = await self.redis.get(cache_key)

        if cached_data:
            messages = json.loads(cached_data)
            return messages[-limit:] if len(messages) > limit else messages

        # Fallback to database
        return await self._get_messages_from_db(session_id, limit)

    async def _update_conversation_state(self,
                                       session_id: str,
                                       message: ChatMessage) -> None:
        """Update conversation state based on new message"""
        state_key = f"chat:state:{session_id}"

        # Get current state
        cached_state = await self.redis.get(state_key)
        if cached_state:
            state_data = json.loads(cached_state)
            state = ConversationState(**state_data)
        else:
            state = ConversationState(session_id=session_id)

        # Update state based on message
        if message.role == MessageRole.USER:
            # Extract entities and intent
            entities = await self.text_processor.extract_entities(message.content)
            intent = await self.text_processor.classify_intent(message.content)

            # Update state
            state.current_entities.update(entities)
            state.current_intent = intent
            state.intent_history.append(intent)
            state.conversation_depth += 1
            state.updated_at = datetime.utcnow()

            # Detect topic changes
            if await self._detect_topic_change(message.content, state):
                await self._handle_topic_change(session_id, state)

        # Cache updated state
        await self.redis.setex(
            state_key,
            self.cache_ttl,
            state.model_dump_json()
        )

    async def _detect_topic_change(self, message_content: str, state: ConversationState) -> bool:
        """Detect if conversation topic changed"""
        return self.text_processor.detect_topic_change(message_content, state.current_topic)

    async def _handle_topic_change(self, session_id: str, state: ConversationState) -> None:
        """Handle topic change by updating state"""
        new_topic = await self.text_processor.extract_topic(state.current_entities.get("topic", ""))
        if new_topic:
            state.current_topic = new_topic

    async def _trigger_summarization(self, session_id: str) -> None:
        """Trigger summarization of older messages"""
        try:
            # Get messages that need summarization
            messages_to_summarize = await self._get_messages_for_summarization(session_id)

            if len(messages_to_summarize) >= self.summary_threshold:
                # Generate summary
                summary = await self.text_processor.summarize_conversation(
                    messages_to_summarize
                )

                # Store in long-term memory
                await self._store_long_term_memory(session_id, summary, messages_to_summarize)

                # Clean up old messages from cache
                await self._cleanup_summarized_messages(session_id, messages_to_summarize)

        except Exception as e:
            logger.error(f"Error in summarization: {e}")

    async def _get_messages_for_summarization(self, session_id: str) -> List[Dict[str, Any]]:
        """Get messages that need to be summarized"""
        # This would typically get older messages from database
        # For now, return empty list
        return []

    async def _get_relevant_memories(self,
                                   user_id: str,
                                   state: ConversationState) -> List[Dict[str, Any]]:
        """Retrieve relevant long-term memories using vector search"""
        try:
            if not state or not state.current_topic:
                return []

            # Skip Weaviate operations for now since client is None
            if self.weaviate is None:
                return []

            # Search Weaviate for relevant memories
            where_filter = {
                "path": ["userId"],
                "operator": "Equal",
                "valueString": user_id
            }

            result = self.weaviate.query\
                .get("ConversationMemory", ["topic", "summary", "keyEntities", "importanceScore"])\
                .with_where(where_filter)\
                .with_near_text({"concepts": [state.current_topic]})\
                .with_limit(5)\
                .do()

            memories = result.get("data", {}).get("Get", {}).get("ConversationMemory", [])

            # Update access counts
            for memory in memories:
                await self._update_memory_access(memory.get("id"))

            return memories

        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")
            return []

    async def _update_memory_access(self, memory_id: str) -> None:
        """Update access count for a memory"""
        # This would update the access count in the database
        pass

    async def _build_context_summary(self,
                                   recent_messages: List[Dict[str, Any]],
                                   long_term_memories: List[Dict[str, Any]],
                                   state: ConversationState) -> str:
        """Build a comprehensive context summary"""
        try:
            summary_parts = []

            # Current conversation context
            if state and state.current_topic:
                summary_parts.append(f"Current topic: {state.current_topic}")

            if state and state.current_entities:
                entities_str = ", ".join([f"{k}: {v}" for k, v in state.current_entities.items()])
                summary_parts.append(f"Key entities: {entities_str}")

            # Recent conversation flow
            if recent_messages:
                recent_topics = self._extract_topics_from_messages(recent_messages)
                if recent_topics:
                    summary_parts.append(f"Recent discussion: {', '.join(recent_topics)}")

            # Relevant background from long-term memory
            if long_term_memories:
                memory_topics = [mem.get("topic", "") for mem in long_term_memories]
                summary_parts.append(f"Related past discussions: {', '.join(memory_topics)}")

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Error building context summary: {e}")
            return "Unable to build context summary"

    def _extract_topics_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract topics from recent messages"""
        topics = []
        for msg in messages:
            entities = msg.get("entities", {})
            for entity_type, entity_value in entities.items():
                if entity_type in ["organization", "technology", "concept"]:
                    topics.append(str(entity_value))
        return list(set(topics))  # Remove duplicates

    async def _store_long_term_memory(self,
                                    session_id: str,
                                    summary: str,
                                    messages: List[Dict[str, Any]]) -> None:
        """Store summarized conversation in Weaviate"""
        try:
            # Skip Weaviate operations for now since client is None
            if self.weaviate is None:
                logger.info("Skipping long-term memory storage (Weaviate client not available)")
                return

            # Extract key information
            entities = {}
            topics = []

            for msg in messages:
                if msg.get("entities"):
                    entities.update(msg["entities"])
                # Extract topics using text processing
                topic = await self.text_processor.extract_topic(msg.get("content", ""))
                if topic:
                    topics.append(topic)

            # Create memory object
            memory_data = {
                "sessionId": session_id,
                "topic": ", ".join(set(topics)) if topics else "General Discussion",
                "summary": summary,
                "keyEntities": entities,
                "importanceScore": self._calculate_importance_score(messages),
                "messageCount": len(messages),
                "createdAt": datetime.utcnow().isoformat()
            }

            # Store in Weaviate
            self.weaviate.data_object.create(
                data_object=memory_data,
                class_name="ConversationMemory"
            )

        except Exception as e:
            logger.error(f"Error storing long-term memory: {e}")

    def _calculate_importance_score(self, messages: List[Dict[str, Any]]) -> float:
        """Calculate importance score for memory prioritization"""
        score = 0.0

        # Factors that increase importance:
        # - Length of conversation
        score += len(messages) * 0.1

        # - Presence of entities
        entity_count = sum(len(msg.get("entities", {})) for msg in messages)
        score += entity_count * 0.2

        # - Specific intents (definitions, improvements are more important)
        important_intents = ["definition", "improvement", "explanation"]
        for msg in messages:
            if msg.get("intent") in important_intents:
                score += 0.5

        return min(score, 10.0)  # Cap at 10.0

    async def _persist_message(self, message: ChatMessage) -> None:
        """Persist message to database"""
        # This would save the message to your database
        # For now, we'll skip actual database operations
        pass

    async def _get_message_count(self, session_id: str) -> int:
        """Get total message count for session"""
        cache_key = f"chat:session:{session_id}"
        cached_data = await self.redis.get(cache_key)

        if cached_data:
            messages = json.loads(cached_data)
            return len(messages)

        return 0

    async def _queue_for_summarization(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Queue messages for summarization"""
        # This could queue messages for background summarization
        pass

    async def _cleanup_summarized_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Clean up messages that have been summarized"""
        # This would remove old messages that have been summarized
        pass

    async def _get_messages_from_db(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback to get messages from database"""
        # This would query the database for messages
        return []

    async def _get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """Get conversation state from cache or database"""
        state_key = f"chat:state:{session_id}"

        cached_state = await self.redis.get(state_key)
        if cached_state:
            state_data = json.loads(cached_state)
            return ConversationState(**state_data)

        return None

    async def save_session(self, session: ChatSession) -> None:
        """Save complete session to database"""
        # Implementation for database persistence
        pass

    async def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from database"""
        # Implementation for database loading
        pass

    async def clear_session_memory(self, session_id: str) -> None:
        """Clear all memory for a session"""
        cache_key = f"chat:session:{session_id}"
        state_key = f"chat:state:{session_id}"

        await self.redis.delete(cache_key)
        await self.redis.delete(state_key)
