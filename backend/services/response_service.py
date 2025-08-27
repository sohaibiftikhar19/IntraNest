#!/usr/bin/env python3
"""
Professional response generation service for IntraNest 2.0
"""

import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class ProfessionalResponseGenerator:
    """Enterprise-grade response generator"""

    def analyze_user_intent(self, message: str) -> Dict:
        """Analyze user intent from message"""
        message_lower = message.lower().strip()

        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(word in message_lower for word in greeting_words) and len(message.split()) <= 3:
            return {"type": "greeting", "confidence": 0.95}

        help_patterns = ["help", "what can you do", "capabilities", "features", "how to use", "what do you do"]
        if any(phrase in message_lower for phrase in help_patterns):
            return {"type": "help", "confidence": 0.9}

        return {"type": "contextual", "confidence": 0.7}

    def get_time_appropriate_greeting(self) -> str:
        """Get time-appropriate greeting"""
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            return "Good morning"
        elif 12 <= current_hour < 17:
            return "Good afternoon"
        elif 17 <= current_hour < 22:
            return "Good evening"
        else:
            return "Hello"

    def generate_greeting_response(self) -> str:
        """Professional greeting with time awareness"""
        greeting = self.get_time_appropriate_greeting()
        return f"""{greeting}! I'm IntraNest AI, your enterprise knowledge assistant.

I can help you with document analysis, knowledge search, data insights, and content creation. All responses include source citations when referencing your organizational knowledge.

What would you like to work on today?"""

    def generate_help_response(self) -> str:
        """Professional capabilities overview"""
        return """**IntraNest AI Capabilities**

I can assist you with:

**📄 Document Analysis**
- Summarize reports, policies, and technical documents
- Extract key insights and action items
- Compare multiple documents using advanced RAG

**🔍 Knowledge Search**
- Find specific information across your knowledge base
- Answer questions with precise source citations
- Provide contextual explanations using semantic search

**📊 Data Insights**
- Analyze trends and patterns in your documents
- Generate executive summaries with source attribution
- Create actionable recommendations based on your content

What specific task would you like assistance with?"""

    async def generate_professional_response(self, user_message: str, user_id: str = "anonymous", model: str = "IntraNest-AI", rag_service=None) -> str:
        """Generate enterprise-grade responses"""
        try:
            if not user_message or user_message.strip() == "":
                user_message = "Hello"

            intent = self.analyze_user_intent(user_message)
            logger.debug(f"🎯 Intent detected: {intent['type']}")

            # For contextual queries, try RAG first
            if intent["type"] == "contextual" and rag_service:
                rag_result = await rag_service.generate_rag_response(user_message, user_id)
                if rag_result["has_context"]:
                    logger.info("✅ Using RAG response with document context")
                    return rag_result["response"]

            # Fall back to standard responses
            if intent["type"] == "greeting":
                return self.generate_greeting_response()
            elif intent["type"] == "help":
                return self.generate_help_response()
            else:
                return f"I can help you with that topic. Please upload some documents so I can provide specific information based on your content."

        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."
