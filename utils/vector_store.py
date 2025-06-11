import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from pymilvus import DataType, MilvusClient, model

from utils.logger import setup_logger

logger = setup_logger()


class SekaiVectorStore:
    """Sekai story vector store - Milvus implementation"""

    def __init__(
        self,
        embedding_provider: str = "huggingface",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "stories",
    ):
        """
        initialize Milvus vector store

        Args:
            embedding_provider: embedding provider ("openai", "huggingface")
            embedding_model: model name
            collection_name: Milvus collection name
        """

        self.embedding_provider = embedding_provider.lower()
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # initialize embedding model
        self.embeddings = self._init_embeddings()
        self.dim = self._get_embedding_dimension()

        # connect to Milvus
        self.client = self._connect_milvus()

        # initialize collection
        self._init_collection()

        logger.important("✅ Milvus vector store initialized successfully")

    def _init_embeddings(self):
        """initialize embedding model"""

        if self.embedding_provider == "milvus":
            sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model, device="cpu"
            )
            return sentence_transformer_ef
        else:
            raise ValueError(f"unsupported embedding provider: {self.embedding_provider}")

    def _get_embedding_dimension(self) -> int:
        """get embedding dimension"""
        try:
            test_embedding = self.embeddings.encode_documents(["test"])[0]
            return len(test_embedding)
        except Exception as e:
            return 384

    def _connect_milvus(self) -> MilvusClient:
        """connect to Milvus server"""
        try:
            # read configuration from environment variables first
            milvus_uri = os.getenv("MILVUS_URI")
            milvus_token = os.getenv("MILVUS_TOKEN")

            if milvus_uri:
                # use cloud or remote Milvus
                if milvus_token:
                    client = MilvusClient(uri=milvus_uri, token=milvus_token)
                else:
                    client = MilvusClient(uri=milvus_uri)
            else:
                # fallback to local connection
                local_uri = "http://localhost:19530"
                client = MilvusClient(uri=local_uri, token="root:Milvus")

            return client

        except Exception as e:
            logger.error(f"❌ failed to connect to Milvus: {e}")
            raise e

    def _init_collection(self):
        """initialize Milvus collection"""
        # check if collection exists, if so, delete and rebuild (optional)
        if self.client.has_collection(self.collection_name):
            pass
        else:
            # create collection schema
            schema = self.client.create_schema()

            # add fields
            schema.add_field(
                "id",
                DataType.INT64,
                is_primary=True,
                auto_id=False,
                description="story ID",
            )
            schema.add_field(
                "title", DataType.VARCHAR, max_length=500, description="story title"
            )
            schema.add_field(
                "intro", DataType.VARCHAR, max_length=2000, description="story intro"
            )
            schema.add_field("tags", DataType.JSON, description="story tags")
            schema.add_field(
                "created_at", DataType.VARCHAR, max_length=100, description="created time"
            )
            schema.add_field(
                "dense_content_vector",
                DataType.FLOAT_VECTOR,
                dim=self.dim,
                description="content vector",
            )

            # prepare index parameters
            index_params = self.client.prepare_index_params()
            index_params.add_index("dense_content_vector", metric_type="COSINE")

            # create collection (auto-load)
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            pass

    def add_story(self, story_data: Dict[str, Any]) -> bool:
        """
        add single story to vector store

        Args:
            story_data: story data dictionary, containing id, title, intro, tags, etc.

        Returns:
            if success, return True, otherwise return False
        """
        try:
            content = f"Title: {story_data['title']}\n\nStory: {story_data['intro']}\n\nLabel: {', '.join(story_data.get('tags', []))}"

            vector = self.embeddings.encode_documents([content])[0]
            row = {
                "id": story_data["id"],
                "title": story_data["title"],
                "intro": story_data["intro"],
                "tags": story_data.get("tags", []),
                "created_at": story_data.get("created_at", ""),
                "dense_content_vector": vector,
            }

            self.client.insert(self.collection_name, [row])
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add story: {e}")
            return False

    def similarity_search_by_query(
        self,
        query: str,
        user_tags: List[str] = None,
        k: int = 20,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search based on query

        Args:
            query: Search query
            user_tags: User preference tags (for query enhancement)
            k: Number of results to return
            min_similarity: Minimum similarity threshold (not yet implemented)

        Returns:
            List of (document, similarity score) tuples
        """
        try:
            enhanced_query = query
            if user_tags:
                tags_text = ", ".join(user_tags)
                enhanced_query = f"{query}\n\nUser preference: {tags_text}"

            query_vector = self.embeddings.encode_queries([enhanced_query])[0]

            search_params = {
                "metric_type": "COSINE",
            }

            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=k,
                search_params=search_params,
                anns_field="dense_content_vector",
                output_fields=["id", "title", "intro", "tags", "created_at"],
            )

            documents_with_scores = []
            for hit in results[0]:
                similarity = 1 - hit["distance"]
                doc = Document(
                    page_content=f"Title: {hit['entity']['title']}\n\nStory: {hit['entity']['intro']}",
                    metadata={
                        "story_id": int(hit["entity"]["id"]),
                        "title": hit["entity"]["title"],
                        "intro": hit["entity"]["intro"],
                        "tags": hit["entity"].get("tags", []),
                        "created_at": hit["entity"].get("created_at"),
                        "source": "milvus",
                    },
                )
                documents_with_scores.append((doc, similarity))

            return documents_with_scores

        except Exception as e:
            logger.error(f"❌ Failed to search vector: {e}")
            return []

    def get_story_by_id(self, story_id: int) -> Optional[Document]:
        """Get Story By ID"""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f"id == {story_id}",
                output_fields=["id", "title", "intro", "tags", "created_at"],
            )

            if results:
                story = results[0]
                doc = Document(
                    page_content=f"Title: {story['title']}\n\nStory: {story['intro']}",
                    metadata={
                        "story_id": story["id"],
                        "title": story["title"],
                        "intro": story["intro"],
                        "tags": story.get("tags", []),
                        "created_at": story.get("created_at"),
                        "source": "milvus",
                    },
                )
                return doc
            return None

        except Exception as e:
            logger.error(f"❌ Failed to get story (ID: {story_id}): {e}")
            return None

    def drop_collection(self):
        try:
            self.client.drop_collection(self.collection_name)
        except Exception as e:
            logger.error(f"❌ Failed to drop collection: {e}")


# global vector store instance
_global_vector_store = None


def get_vector_store(
    embedding_provider: str = "milvus",
    embedding_model: str = "all-MiniLM-L6-v2",
    **kwargs,
) -> SekaiVectorStore:
    """get global vector store instance"""
    global _global_vector_store

    if _global_vector_store is None:
        _global_vector_store = SekaiVectorStore(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            **kwargs,
        )

    return _global_vector_store
