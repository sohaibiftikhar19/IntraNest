# services/query_rewriter.py
import openai
from typing import List, Dict, Optional, Any
from models.conversation_models import (
    ChatMessage, ConversationState, QueryRewriteRequest, QueryRewriteResponse
)
import re
import logging

logger = logging.getLogger(__name__)

class QueryRewriter:
    """Handles query rewriting and coreference resolution for conversational RAG"""
    
    def __init__(self, openai_client: openai.AsyncOpenAI, config: Dict[str, Any]):
        self.openai = openai_client
        self.config = config
        self.model = config.get("rewrite_model", "gpt-3.5-turbo")
        self.max_context_messages = config.get("max_context_messages", 5)
    
    async def rewrite_query_with_context(self, request: QueryRewriteRequest) -> QueryRewriteResponse:
        """Main method to rewrite queries incorporating conversation context"""
        try:
            # 1. Analyze query for coreferences
            coreferences = self._detect_coreferences(request.original_query)
            
            # 2. Extract relevant context
            context = self._extract_relevant_context(
                request.conversation_context, 
                request.current_state
            )
            
            # 3. Rewrite query using LLM
            rewritten_query = await self._llm_rewrite_query(
                request.original_query,
                context,
                coreferences
            )
            
            # 4. Resolve entities
            resolved_entities = self._resolve_entities(
                request.original_query,
                rewritten_query,
                request.current_state
            )
            
            # 5. Calculate confidence
            confidence = self._calculate_confidence(
                request.original_query,
                rewritten_query,
                context
            )
            
            return QueryRewriteResponse(
                original_query=request.original_query,
                rewritten_query=rewritten_query,
                resolved_entities=resolved_entities,
                confidence_score=confidence,
                reasoning=f"Resolved {len(resolved_entities)} entities from context"
            )
            
        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            # Fallback: return original query
            return QueryRewriteResponse(
                original_query=request.original_query,
                rewritten_query=request.original_query,
                confidence_score=0.0,
                reasoning=f"Rewriting failed: {str(e)}"
            )
    
    def _detect_coreferences(self, query: str) -> List[Dict[str, Any]]:
        """Detect pronouns and references that need resolution"""
        coreferences = []
        
        # Common pronouns and references
        pronouns = {
            'it': 'pronoun',
            'they': 'pronoun', 
            'them': 'pronoun',
            'this': 'demonstrative',
            'that': 'demonstrative',
            'these': 'demonstrative',
            'those': 'demonstrative'
        }
        
        # Possessive pronouns
        possessive = {
            'its': 'possessive',
            'their': 'possessive',
            'his': 'possessive',
            'her': 'possessive'
        }
        
        # Definite references
        definite_patterns = [
            r'\bthe ([\w\s]+) (capabilities?|features?|benefits?|solutions?)',
            r'\bthe (above|mentioned|discussed) ([\w\s]+)',
            r'\bthe (AI|artificial intelligence|system|platform|solution)'
        ]
        
        words = query.lower().split()
        
        # Check for pronouns and possessives
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w]', '', word)
            if word_clean in pronouns:
                coreferences.append({
                    'text': word,
                    'type': pronouns[word_clean],
                    'position': i,
                    'needs_resolution': True
                })
            elif word_clean in possessive:
                coreferences.append({
                    'text': word,
                    'type': possessive[word_clean],
                    'position': i,
                    'needs_resolution': True
                })
        
        # Check for definite references
        for pattern in definite_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                coreferences.append({
                    'text': match.group(0),
                    'type': 'definite_reference',
                    'position': match.start(),
                    'needs_resolution': True
                })
        
        return coreferences
    
    def _extract_relevant_context(self, 
                                messages: List[ChatMessage], 
                                state: ConversationState) -> Dict[str, Any]:
        """Extract relevant context for query rewriting"""
        context = {
            'recent_topics': [],
            'entities': state.current_entities.copy() if state else {},
            'last_user_query': None,
            'last_assistant_response': None,
            'conversation_flow': []
        }
        
        # Get recent messages (limit to avoid token overflow)
        recent_messages = messages[-self.max_context_messages:] if messages else []
        
        for msg in recent_messages:
            context['conversation_flow'].append({
                'role': msg.role.value,
                'content': msg.content[:200],  # Truncate for efficiency
                'entities': msg.entities
            })
            
            if msg.role.value == 'user':
                context['last_user_query'] = msg.content
            elif msg.role.value == 'assistant':
                context['last_assistant_response'] = msg.content[:300]
        
        # Extract topics from recent messages
        if state and state.current_topic:
            context['recent_topics'].append(state.current_topic)
        
        return context
    
    async def _llm_rewrite_query(self, 
                               original_query: str,
                               context: Dict[str, Any],
                               coreferences: List[Dict[str, Any]]) -> str:
        """Use LLM to rewrite query with context"""
        
        # Build context string
        context_parts = []
        
        if context.get('entities'):
            entities_str = ', '.join([f"{k}: {v}" for k, v in context['entities'].items()])
            context_parts.append(f"Key entities in conversation: {entities_str}")
        
        if context.get('last_user_query'):
            context_parts.append(f"Previous user question: {context['last_user_query']}")
        
        if context.get('last_assistant_response'):
            context_parts.append(f"Previous assistant response: {context['last_assistant_response']}")
        
        if context.get('recent_topics'):
            context_parts.append(f"Current topics: {', '.join(context['recent_topics'])}")
        
        context_string = '\n'.join(context_parts)
        
        # Build coreference information
        coreference_info = ""
        if coreferences:
            coreference_texts = [ref['text'] for ref in coreferences]
            coreference_info = f"\nReferences to resolve: {', '.join(coreference_texts)}"
        
        system_prompt = """You are a query rewriting assistant. Your job is to rewrite user queries to be complete and standalone by incorporating relevant context from the conversation.

Rules:
1. Replace pronouns (it, they, this, that) with specific entities from context
2. Resolve definite references ("the AI capabilities" → "TCS's AI capabilities")  
3. Make the query self-contained and clear
4. Preserve the user's original intent
5. Don't add information not supported by context
6. If no context is needed, return the original query unchanged

Focus on making queries retrievable and specific."""

        user_prompt = f"""Context from conversation:
{context_string}

Original user query: "{original_query}"
{coreference_info}

Rewrite this query to be complete and standalone, resolving any pronouns or references using the conversation context. Return only the rewritten query."""

        try:
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            rewritten = response.choices[0].message.content.strip()
            
            # Clean up the response (remove quotes if present)
            rewritten = re.sub(r'^["\']|["\']$', '', rewritten)
            
            return rewritten
            
        except Exception as e:
            logger.error(f"LLM rewriting failed: {e}")
            return original_query
    
    def _resolve_entities(self, 
                         original_query: str,
                         rewritten_query: str,
                         state: ConversationState) -> Dict[str, str]:
        """Track what entities were resolved during rewriting"""
        resolved = {}
        
        if not state or not state.current_entities:
            return resolved
        
        # Simple approach: check if entities from state appear in rewritten but not original
        for entity_key, entity_value in state.current_entities.items():
            entity_str = str(entity_value).lower()
            
            # Check if entity appears in rewritten query but not in original
            if (entity_str in rewritten_query.lower() and 
                entity_str not in original_query.lower()):
                resolved[entity_key] = entity_str
        
        return resolved
    
    def _calculate_confidence(self, 
                            original_query: str,
                            rewritten_query: str,
                            context: Dict[str, Any]) -> float:
        """Calculate confidence score for the rewriting"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if query was significantly modified
        if len(rewritten_query) > len(original_query) * 1.2:
            confidence += 0.2
        
        # Higher confidence if context entities are present
        if context.get('entities'):
            for entity_value in context['entities'].values():
                if str(entity_value).lower() in rewritten_query.lower():
                    confidence += 0.1
        
        # Lower confidence if query unchanged
        if original_query.lower().strip() == rewritten_query.lower().strip():
            confidence = 0.3
        
        return min(confidence, 1.0)
    
    async def expand_query_with_synonyms(self, query: str) -> List[str]:
        """Generate query variations with synonyms for improved retrieval"""
        try:
            system_prompt = """Generate 3 alternative phrasings of the given query using synonyms and related terms. Keep the same intent but vary the wording for better document retrieval.

Return only the alternative queries, one per line."""
            
            response = await self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            return [v.strip() for v in variations if v.strip()]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]  # Fallback to original
