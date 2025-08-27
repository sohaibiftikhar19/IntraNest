#!/usr/bin/env python3

# Read main.py and fix the ServiceContext deprecation
with open('main.py', 'r') as f:
    content = f.read()

# Replace ServiceContext usage with Settings
old_service_context = '''            # Create service context (CRITICAL: this ensures consistency)
            from llama_index.core import ServiceContext
            self.service_context = ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model
            )'''

new_settings = '''            # FIXED: Use Settings instead of deprecated ServiceContext
            from llama_index.core import Settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            self.service_context = None  # Not needed with Settings'''

content = content.replace(old_service_context, new_settings)

# Fix VectorStoreIndex.from_vector_store calls to not use service_context
old_index_creation = '''                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    service_context=self.service_context,
                    storage_context=storage_context
                )'''

new_index_creation = '''                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    storage_context=storage_context
                )'''

content = content.replace(old_index_creation, new_index_creation)

# Fix fallback index creation
old_fallback = '''                self.index = VectorStoreIndex(
                    nodes=[],
                    service_context=self.service_context,
                    storage_context=storage_context
                )'''

new_fallback = '''                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context
                )'''

content = content.replace(old_fallback, new_fallback)

# Write the fixed file
with open('main.py', 'w') as f:
    f.write(content)

print("✅ Fixed ServiceContext deprecation issue")
