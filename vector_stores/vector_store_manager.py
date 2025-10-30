from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """
    Manages different types of vector stores.
    FAISS is local, Pinecone is cloud
    """

    def __init__(self):
        """
        create - PineconeVectorStore.from_documents
            - Creates a new index and populates it with documents
            - Used when you want to create a vector store from scratch with your documents
            - Handles embedding creation and upsert operations
            - Typically used during initial setup or when adding new data

        load - PineconeVectorStore (constructor)
            - Connects to an existing index that's already populated
            - Assumes the Pinecone index already exists and contains vectors
            - Used for querying/searching against an existing index
            - More like "connect" than "load" since Pinecone is cloud-based
        """
        self.vector_store_configs = {
            'faiss': {
                'create': FAISS.from_documents,
                'load': FAISS.load_local,
            },
            'pinecone': {
                'create': PineconeVectorStore.from_documents,
                'load': PineconeVectorStore,    # Pinecone doesn't have a load_local equivalent
            }
        }

    def get_vector_store(self, store_name: str, store_type: str, *args, **kwargs):
        """
        Get a vector store instance by performing the specified operation.

        Args:
            store_name: The name of the vector store type ('faiss' or 'pinecone')
            store_type: The operation to perform ('load' or 'create')
            *args: Positional arguments passed to the underlying vector store method
            **kwargs: Keyword arguments passed to the underlying vector store method

        Returns:
            VectorStore: An instance of the requested vector store
        """
        if store_name not in self.vector_store_configs:
            raise ValueError(f"Unsupported vector store type: {store_name}")

        config = self.vector_store_configs[store_name]
        callable_object = config.get(store_type)

        return callable_object(*args, **kwargs)
