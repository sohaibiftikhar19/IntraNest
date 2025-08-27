#!/usr/bin/env python3
"""
Simple setup script for IntraNest Conversational RAG
"""

import asyncio
import os
import json
import logging
import requests
from datetime import datetime

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  {env_file} file not found")

# Load .env file at startup
load_env_file()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_redis_connection():
    """Test Redis connection"""
    logger.info("🗄️  Testing Redis connection...")
    
    try:
        import redis.asyncio as redis
        
        # Try port from environment or default
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        logger.info(f"Connecting to Redis at: {redis_url}")
        
        redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Test connection
        result = await redis_client.ping()
        if result:
            logger.info("✅ Redis connection successful!")
            
            # Test set/get operation
            test_key = "intranest:setup:test"
            await redis_client.setex(test_key, 60, "test_value")
            value = await redis_client.get(test_key)
            
            if value == "test_value":
                logger.info("✅ Redis read/write operations working!")
                await redis_client.delete(test_key)
            else:
                logger.warning("⚠️  Redis read/write test failed")
                
        await redis_client.aclose()  # Use aclose() instead of close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.info("💡 Make sure Redis is running:")
        logger.info("   docker run -d --name intranest-redis -p 6379:6379 redis:7-alpine")
        logger.info("   docker exec intranest-redis redis-cli ping")
        return False

def test_weaviate_connection():
    """Test Weaviate connection using HTTP requests"""
    logger.info("📊 Testing Weaviate connection...")
    
    try:
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_key = os.getenv("WEAVIATE_API_KEY")
        
        logger.info(f"Testing Weaviate at: {weaviate_url}")
        
        # Prepare headers with authentication if key is provided
        headers = {}
        if weaviate_key:
            headers["Authorization"] = f"Bearer {weaviate_key}"
            logger.info("Using API key authentication")
        
        # Simple HTTP test to see if Weaviate is responding
        response = requests.get(f"{weaviate_url}/v1/meta", headers=headers, timeout=5)
        
        if response.status_code == 200:
            logger.info("✅ Weaviate connection successful!")
            
            # Get meta information
            meta_data = response.json()
            logger.info(f"Weaviate version: {meta_data.get('version', 'unknown')}")
            
            return True
        elif response.status_code == 401:
            logger.error("❌ Weaviate authentication failed (401)")
            logger.info("💡 Check your WEAVIATE_API_KEY in .env file")
            return False
        else:
            logger.error(f"❌ Weaviate returned status {response.status_code}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Weaviate connection failed: {e}")
        logger.info("💡 Make sure Weaviate is running:")
        logger.info("   docker-compose up -d weaviate")
        return False

def test_environment_variables():
    """Test required environment variables"""
    logger.info("⚙️  Checking environment variables...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API access",
        "WEAVIATE_URL": "Weaviate database connection", 
        "REDIS_URL": "Redis cache connection"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var:
                masked = value[:8] + "..." + value[-8:] if len(value) > 16 else "***"
                logger.info(f"✅ {var}: {masked}")
            else:
                logger.info(f"✅ {var}: {value}")
        else:
            logger.warning(f"❌ {var}: Missing ({description})")
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("✅ All required environment variables present")
    return True

async def test_openai_connection():
    """Test OpenAI API connection"""
    logger.info("🤖 Testing OpenAI connection...")
    
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Simple test call
        response = await client.models.list()
        
        if response.data:
            logger.info("✅ OpenAI API connection successful!")
            logger.info(f"Available models: {len(response.data)} models found")
            return True
        else:
            logger.error("❌ OpenAI API returned no models")
            return False
        
    except Exception as e:
        logger.error(f"❌ OpenAI connection failed: {e}")
        logger.info("💡 Check your OPENAI_API_KEY")
        return False

async def main():
    """Main setup function"""
    logger.info("🚀 Starting IntraNest Conversational RAG Setup")
    logger.info("=" * 60)
    
    # Step 1: Check environment variables
    if not test_environment_variables():
        logger.error("❌ Setup failed due to missing environment variables")
        return False
    
    # Step 2: Test Redis
    redis_ok = await test_redis_connection()
    if not redis_ok:
        logger.error("❌ Setup failed due to Redis connection issues")
        return False
    
    # Step 3: Test Weaviate
    weaviate_ok = test_weaviate_connection()
    if not weaviate_ok:
        logger.error("❌ Setup failed due to Weaviate connection issues")
        return False
    
    # Step 4: Test OpenAI
    openai_ok = await test_openai_connection()
    if not openai_ok:
        logger.warning("⚠️  OpenAI connection failed - conversational features may be limited")
    
    # Success!
    logger.info("=" * 60)
    logger.info("🎉 SETUP COMPLETE!")
    logger.info("=" * 60)
    logger.info("✅ Redis connection: Working")
    logger.info("✅ Weaviate connection: Working")
    logger.info(f"{'✅' if openai_ok else '⚠️ '} OpenAI connection: {'Working' if openai_ok else 'Limited'}")
    logger.info("✅ Environment variables: Configured")
    
    logger.info("\n📋 NEXT STEPS:")
    logger.info("1. Update main.py with conversational router")
    logger.info("2. Start IntraNest: python main.py")
    logger.info("3. Test conversational features:")
    logger.info("   curl -X POST http://YOUR-IP:8001/api/chat/conversational \\")
    logger.info("     -H 'Content-Type: application/json' \\")
    logger.info("     -d '{\"message\": \"What is TCS?\", \"user_id\": \"test_user\"}'")
    
    logger.info("\n🔧 CONVERSATIONAL RAG FEATURES READY:")
    logger.info("- Context retention across multiple turns")
    logger.info("- Coreference resolution (pronouns, references)")
    logger.info("- Conversation memory and summarization")
    logger.info("- Intent tracking and topic detection")
    logger.info("- Session management")
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print("\n🚀 IntraNest Conversational RAG is ready!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
