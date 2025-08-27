#!/usr/bin/env python3

# Add debug logging to the initialization
with open('main.py', 'r') as f:
    content = f.read()

# Find the initialize_services method and add more logging
old_init = '''    def initialize_services(self):
        """Initialize LlamaIndex services with correct Weaviate configuration"""
        try:
            if not LLAMAINDEX_AVAILABLE or not llamaindex_components:
                logger.error("❌ LlamaIndex components not available")
                return'''

new_init = '''    def initialize_services(self):
        """Initialize LlamaIndex services with correct Weaviate configuration"""
        try:
            logger.info("🚀 DEBUG: Starting LlamaIndex initialization...")
            logger.info(f"🔍 DEBUG: LLAMAINDEX_AVAILABLE = {LLAMAINDEX_AVAILABLE}")
            logger.info(f"🔍 DEBUG: llamaindex_components keys = {list(llamaindex_components.keys()) if 'llamaindex_components' in globals() else 'NOT DEFINED'}")
            
            if not LLAMAINDEX_AVAILABLE or not llamaindex_components:
                logger.error("❌ LlamaIndex components not available")
                return'''

content = content.replace(old_init, new_init)

# Add debug after Weaviate connection
old_weaviate = '''            if not self.weaviate_client.is_ready():
                logger.error("❌ Weaviate client not ready")
                return

            logger.info("✅ Weaviate client connected")'''

new_weaviate = '''            if not self.weaviate_client.is_ready():
                logger.error("❌ Weaviate client not ready")
                return

            logger.info("✅ Weaviate client connected")
            logger.info("🔍 DEBUG: About to initialize LlamaIndex components...")'''

content = content.replace(old_weaviate, new_weaviate)

# Write the debug version
with open('main.py', 'w') as f:
    f.write(content)

print("✅ Added debug logging to initialization")
