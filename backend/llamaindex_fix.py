#!/usr/bin/env python3

# Read the current main.py and fix the query engine issue
with open('main.py', 'r') as f:
    content = f.read()

# Find and replace the problematic query engine setup
old_query_engine = '''            # Create query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact",
                verbose=True
            )'''

new_query_engine = '''            # Create query engine - fix parameter conflict
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )'''

content = content.replace(old_query_engine, new_query_engine)

# Also fix the custom query engine in generate_rag_response
old_custom_query = '''            custom_query_engine = self.index.as_query_engine(
                retriever=retriever,
                response_mode="compact",
                verbose=True
            )'''

new_custom_query = '''            custom_query_engine = self.index.as_query_engine(
                response_mode="compact"
            )'''

content = content.replace(old_custom_query, new_custom_query)

# Write the fixed file
with open('main.py', 'w') as f:
    f.write(content)

print("✅ Fixed LlamaIndex query engine configuration")
