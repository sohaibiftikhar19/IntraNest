#!/usr/bin/env python3

# Read the current main.py
with open('main.py', 'r') as f:
    content = f.read()

# Check if ProfessionalResponseGenerator class exists
if 'class ProfessionalResponseGenerator:' not in content:
    print("❌ ProfessionalResponseGenerator class is missing, restoring it...")
    
    # Find a good place to insert it (before the response_generator instantiation)
    insertion_point = content.find('response_generator = ProfessionalResponseGenerator()')
    
    if insertion_point != -1:
        # Insert the class definition before the instantiation
        professional_response_class = '''
# Professional Response Generator (keeping your existing implementation)
class ProfessionalResponseGenerator:
    """Enterprise-grade response generator based on industry best practices"""

    def __init__(self):
        self.style_guide = {
            "avoid_phrases": [
                "I understand you're asking about",
                "I understand that",
                "Let me help you with that",
                "Here's what I found"
            ]
        }

    def analyze_user_intent(self, message: str) -> Dict:
        """Analyze user intent from message"""
        message_lower = message.lower().strip()

        # Greeting detection
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(word in message_lower for word in greeting_words) and len(message.split()) <= 3:
            return {"type": "greeting", "confidence": 0.95}

        # Help request detection
        help_patterns = ["help", "what can you do", "capabilities", "features", "how to use", "what do you do"]
        if any(phrase in message_lower for phrase in help_patterns):
            return {"type": "help", "confidence": 0.9}

        # Explanation request detection
        explanation_patterns = [
            "explain", "what is", "how does", "define", "meaning", "tell me about",
            "what do you mean", "what does", "mean by", "what are", "describe"
        ]
        if any(phrase in message_lower for phrase in explanation_patterns):
            return {"type": "explanation", "confidence": 0.8}

        # Summary request detection
        summary_words = ["summarize", "summary", "key points", "overview", "main points", "sum up"]
        if any(word in message_lower for word in summary_words):
            return {"type": "summary", "confidence": 0.85}

        # Analysis request detection
        analysis_words = ["analyze", "analysis", "compare", "evaluate", "assess", "review"]
        if any(word in message_lower for word in analysis_words):
            return {"type": "analysis", "confidence": 0.85}

        return {"type": "contextual", "confidence": 0.7}

    def extract_topic(self, message: str) -> str:
        """Extract main topic from user message"""
        words = message.split()

        # Remove common question words and articles
        stop_words = {
            "what", "how", "why", "when", "where", "who", "can", "could",
            "would", "should", "is", "are", "the", "a", "an", "do", "does",
            "tell", "me", "about", "explain", "to", "you", "your", "our"
        }

        content_words = [
            word.strip(".,?!").title()
            for word in words
            if word.lower() not in stop_words and len(word) > 2
        ]

        if content_words:
            topic_words = content_words[:3]
            return " ".join(topic_words)

        return "Your Query"

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

**✍️ Content Creation**
- Draft professional communications based on your knowledge
- Create structured reports from multiple sources
- Generate meeting summaries with document references

What specific task would you like assistance with?"""

    async def generate_professional_response(self, user_message: str, user_id: str = "anonymous", model: str = "IntraNest-AI") -> str:
        """Generate enterprise-grade responses with document context when available"""
        try:
            if not user_message or user_message.strip() == "":
                user_message = "Hello"

            logger.info(f"🧠 Generating enhanced response for: {user_message[:100]}...")

            # Analyze user intent
            intent = self.analyze_user_intent(user_message)
            logger.info(f"🎯 Intent detected: {intent['type']} (confidence: {intent['confidence']})")

            # For queries that could benefit from document context, try RAG first
            if intent["type"] in ["summary", "analysis", "explanation", "contextual"] and document_service:
                logger.info(f"🔍 Attempting RAG search for user_id: {user_id}")
                rag_result = await document_service.generate_rag_response(user_message, user_id)

                logger.info(f"📄 RAG result - has_context: {rag_result['has_context']}, sources: {len(rag_result.get('sources', []))}")

                if rag_result["has_context"]:
                    logger.info("✅ Using RAG response with document context")
                    return rag_result["response"]
                else:
                    logger.info("⚠️ No relevant documents found, using fallback response")

            # Fall back to standard professional responses
            logger.info(f"📝 Using standard professional response for intent: {intent['type']}")
            if intent["type"] == "greeting":
                return self.generate_greeting_response()
            elif intent["type"] == "help":
                return self.generate_help_response()
            else:
                return f"I can help you with that topic. Please upload some documents so I can provide specific information based on your content."

        except Exception as e:
            logger.error(f"❌ Enhanced response generation error: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again or contact support for assistance."

'''
        
        # Insert the class before the instantiation
        content = content[:insertion_point] + professional_response_class + '\n' + content[insertion_point:]
        
        # Write the fixed file
        with open('main.py', 'w') as f:
            f.write(content)
        
        print("✅ Restored ProfessionalResponseGenerator class")
    else:
        print("❌ Could not find insertion point for class")
else:
    print("✅ ProfessionalResponseGenerator class already exists")
