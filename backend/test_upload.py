#!/usr/bin/env python3
"""Test script for document upload functionality"""

import requests
import json
import time

# Configuration
API_BASE = "http://localhost:8001"
API_KEY = "intranest_3JLzzwF0rQ7I6ulzx1F1RasuBE_1m0EN6xOLsIp8Q7s"
TEST_USER = "test_user_123"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_presigned_url():
    """Test presigned URL generation"""
    print("🧪 Testing presigned URL generation...")
    
    payload = {
        "filename": "test_document.txt",
        "user_id": TEST_USER,
        "file_size": 1024
    }
    
    response = requests.post(f"{API_BASE}/api/documents/presigned-url", 
                           json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Presigned URL generated: {data['document_id']}")
        return data
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return None

def test_file_upload(presigned_data):
    """Test actual file upload to presigned URL"""
    print("🧪 Testing file upload...")
    
    # Sample file content
    test_content = """IntraNest Test Document
    
This is a test document for the IntraNest document management system.
It contains sample content to test the upload and processing functionality.

Key features being tested:
- File upload via presigned URL
- Document processing and chunking
- Weaviate storage integration
- Background processing with Celery

This document should be processed into multiple chunks for semantic search.
"""
    
    # Upload to presigned URL
    upload_response = requests.put(
        presigned_data["upload_url"],
        data=test_content.encode('utf-8'),
        headers={"Content-Type": "text/plain"}
    )
    
    if upload_response.status_code in [200, 204]:
        print("✅ File uploaded successfully")
        return True
    else:
        print(f"❌ Upload failed: {upload_response.status_code}")
        return False

def test_confirm_upload(document_id):
    """Test upload confirmation and processing"""
    print("🧪 Testing upload confirmation...")
    
    payload = {
        "document_id": document_id,
        "user_id": TEST_USER
    }
    
    response = requests.post(f"{API_BASE}/api/documents/confirm-upload",
                           json=payload, headers=headers)
    
    if response.status_code == 200:
        print("✅ Upload confirmed, processing started")
        return True
    else:
        print(f"❌ Confirmation failed: {response.status_code} - {response.text}")
        return False

def test_processing_status(document_id):
    """Test processing status monitoring"""
    print("🧪 Monitoring processing status...")
    
    max_attempts = 30  # 30 seconds max
    for attempt in range(max_attempts):
        response = requests.get(
            f"{API_BASE}/api/documents/status/{document_id}",
            params={"user_id": TEST_USER},
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            progress = data.get("progress", 0)
            
            print(f"  Status: {status} ({progress}%)")
            
            if status == "completed":
                print(f"✅ Processing completed! Chunks created: {data.get('chunks_created', 0)}")
                return True
            elif status == "error":
                print(f"❌ Processing failed: {data.get('error', 'Unknown error')}")
                return False
            
            time.sleep(1)
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    
    print("⏰ Processing timeout")
    return False

def test_document_list():
    """Test document listing"""
    print("🧪 Testing document list...")
    
    payload = {"user_id": TEST_USER}
    
    response = requests.post(f"{API_BASE}/api/documents/list",
                           json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        docs = data.get("documents", [])
        print(f"✅ Found {len(docs)} documents for user")
        
        if docs:
            doc = docs[0]
            print(f"  Sample document: {doc.get('filename')} ({doc.get('chunks', 0)} chunks)")
        
        return True
    else:
        print(f"❌ List failed: {response.status_code} - {response.text}")
        return False

def main():
    """Run complete test suite"""
    print("🚀 Starting IntraNest Document Management Tests")
    print("=" * 50)
    
    # Test 1: Generate presigned URL
    presigned_data = test_presigned_url()
    if not presigned_data:
        return
    
    document_id = presigned_data["document_id"]
    
    # Test 2: Upload file
    if not test_file_upload(presigned_data):
        return
    
    # Test 3: Confirm upload
    if not test_confirm_upload(document_id):
        return
    
    # Test 4: Monitor processing
    if not test_processing_status(document_id):
        return
    
    # Test 5: List documents
    test_document_list()
    
    print("\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    main()
