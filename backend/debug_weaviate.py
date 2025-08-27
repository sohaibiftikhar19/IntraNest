#!/usr/bin/env python3
"""
Debug script to check Weaviate database content and LlamaIndex integration
"""

import weaviate
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("🔍 IntraNest 2.0 - Weaviate Debug Script")
    print("=" * 50)
    
    # Connection details
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")
    
    print(f"Weaviate URL: {weaviate_url}")
    print(f"API Key: {weaviate_key[:20]}..." if weaviate_key else "No API key")
    
    # Connect to Weaviate
    try:
        if "://" in weaviate_url:
            host = weaviate_url.split("://")[1]
        else:
            host = weaviate_url
            
        if ":" in host:
            host = host.split(":")[0]
            
        auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_key) if weaviate_key else None
        
        client = weaviate.connect_to_local(
            host=host,
            port=8080,
            auth_credentials=auth_config
        )
        
        if not client.is_ready():
            print("❌ Weaviate client not ready")
            return
            
        print("✅ Connected to Weaviate")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Check schema/collections
    print("\n📋 Checking Collections:")
    try:
        collections = client.collections.list_all()
        print(f"Found {len(collections)} collections:")
        
        for name, collection in collections.items():
            print(f"  • {name}")
            
    except Exception as e:
        print(f"❌ Error getting collections: {e}")
    
    # Check Documents collection specifically
    print("\n📄 Checking Documents Collection:")
    try:
        if "Documents" in collections:
            documents_collection = client.collections.get("Documents")
            
            # Get count
            try:
                # Try to get a few objects to check if data exists
                response = documents_collection.query.fetch_objects(limit=5)
                objects = response.objects if response.objects else []
                
                print(f"✅ Documents collection exists with {len(objects)} objects (showing first 5)")
                
                # Show sample objects
                for i, obj in enumerate(objects):
                    print(f"\n  Object {i+1}:")
                    print(f"    UUID: {obj.uuid}")
                    if hasattr(obj, 'properties'):
                        props = obj.properties
                        print(f"    Filename: {props.get('filename', 'N/A')}")
                        print(f"    User ID: {props.get('user_id', 'N/A')}")
                        print(f"    Content: {props.get('content', 'N/A')[:100]}...")
                    else:
                        print(f"    Properties: {obj}")
                        
            except Exception as e:
                print(f"❌ Error querying Documents: {e}")
                
        else:
            print("❌ Documents collection does not exist")
            
    except Exception as e:
        print(f"❌ Error checking Documents collection: {e}")
    
    # Try direct search
    print("\n🔍 Testing Direct Search:")
    try:
        if "Documents" in collections:
            documents_collection = client.collections.get("Documents")
            
            # Try a simple text search
            response = documents_collection.query.near_text(
                query="test",
                limit=3
            )
            
            print(f"Found {len(response.objects)} results for 'test' query")
            
            for i, obj in enumerate(response.objects):
                props = obj.properties if hasattr(obj, 'properties') else {}
                print(f"  Result {i+1}: {props.get('filename', 'N/A')} (user: {props.get('user_id', 'N/A')})")
                
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    # Test with user filter
    print("\n👤 Testing User-Filtered Search:")
    try:
        if "Documents" in collections:
            documents_collection = client.collections.get("Documents")
            
            # Search with user filter
            from weaviate.classes.query import Filter
            
            response = documents_collection.query.near_text(
                query="test",
                where=Filter.by_property("user_id").equal("test123"),
                limit=3
            )
            
            print(f"Found {len(response.objects)} results for user 'test123'")
            
            for i, obj in enumerate(response.objects):
                props = obj.properties if hasattr(obj, 'properties') else {}
                print(f"  Result {i+1}: {props.get('filename', 'N/A')} (user: {props.get('user_id', 'N/A')})")
                
    except Exception as e:
        print(f"❌ User-filtered search error: {e}")
    
    # Close connection
    client.close()
    print("\n✅ Debug complete")

if __name__ == "__main__":
    main()
