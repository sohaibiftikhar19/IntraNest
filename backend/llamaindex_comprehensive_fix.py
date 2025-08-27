#!/usr/bin/env python3

# Read the current main.py
with open('main.py', 'r') as f:
    content = f.read()

# Replace the entire search_documents method with a more robust version
old_search_method = '''    async def search_documents(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search documents using LlamaIndex with user filtering"""
        try:
            if not self.index:
                logger.warning("⚠️ LlamaIndex not available for search")
                return []

            # Use LlamaIndex retriever with filtering - fix the Filter import issue
            try:
                from weaviate.classes.query import Filter
                user_filter = Filter.by_property("user_id").equal(user_id)
            except:
                # Fallback if Filter import fails
                user_filter = None
                logger.warning("⚠️ Weaviate Filter not available, searching without user filter")

            # Create retriever with proper parameters
            if user_filter:
                retriever = self.index.as_retriever(
                    similarity_top_k=limit,
                    filters=user_filter
                )
            else:
                retriever = self.index.as_retriever(similarity_top_k=limit)

            # Retrieve relevant nodes
            nodes = retriever.retrieve(query)

            results = []
            for node in nodes:
                # Apply user filtering manually if Weaviate filter didn't work
                if user_filter is None:
                    node_user_id = node.metadata.get("user_id", "")
                    if node_user_id != user_id:
                        continue

                results.append({
                    "content": node.text,
                    "filename": node.metadata.get("filename", "Unknown"),
                    "chunk_id": node.metadata.get("chunk_id", 0),
                    "page_number": node.metadata.get("page_number", 1),
                    "similarity_score": node.score if hasattr(node, 'score') else 0.0,
                    "document_id": node.metadata.get("document_id", ""),
                    "node_id": node.node_id
                })

            return results

        except Exception as e:
            logger.error(f"❌ LlamaIndex search error: {e}")
            return []'''

new_search_method = '''    async def search_documents(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search documents using direct Weaviate access (bypass LlamaIndex retriever issues)"""
        try:
            if not self.weaviate_client:
                logger.warning("⚠️ Weaviate client not available for search")
                return []

            # Use direct Weaviate search to avoid LlamaIndex TextNode validation issues
            documents_collection = self.weaviate_client.collections.get("Documents")

            # Build where filter for user isolation
            from weaviate.classes.query import Filter
            where_filter = Filter.by_property("user_id").equal(user_id)

            # Perform hybrid search (combines vector + keyword search)
            try:
                response = documents_collection.query.hybrid(
                    query=query,
                    limit=limit,
                    alpha=0.7,  # Balance between vector and keyword search
                    where=where_filter
                )
            except Exception as e:
                logger.warning(f"⚠️ Hybrid search failed, trying keyword search: {e}")
                # Fallback to pure keyword search
                response = documents_collection.query.bm25(
                    query=query,
                    limit=limit,
                    where=where_filter
                )

            results = []
            for obj in response.objects:
                # Extract content from either 'content' or '_node_content' fields
                content = obj.properties.get("content", "")
                if not content and "_node_content" in obj.properties:
                    # Try to extract from LlamaIndex node content
                    try:
                        import json
                        node_data = json.loads(obj.properties["_node_content"])
                        content = node_data.get("text", "")
                    except:
                        content = str(obj.properties.get("_node_content", ""))[:200] + "..."

                results.append({
                    "content": content,
                    "filename": obj.properties.get("filename", "Unknown"),
                    "chunk_id": obj.properties.get("chunk_id", 0),
                    "page_number": obj.properties.get("page_number", 1),
                    "similarity_score": getattr(obj.metadata, 'score', 0.8),  # Default score
                    "document_id": obj.properties.get("document_id", ""),
                    "node_id": obj.properties.get("node_id", str(obj.uuid))
                })

            logger.info(f"✅ Found {len(results)} documents for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"❌ Document search error: {e}")
            return []'''

content = content.replace(old_search_method, new_search_method)

# Write the fixed file
with open('main.py', 'w') as f:
    f.write(content)

print("✅ Applied comprehensive LlamaIndex fix")
