from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """ Manages different types of vector stores."""
    def __init__(self):
        self.vector_store_configs = {
            'faiss': {
                'class': FAISS.load_local,
            },
            'pinecone': {
                'class': PineconeVectorStore,
            }
        }

    def get_vector_store(self, store_type, *args, **kwargs):
        """ Factory method to get a vector store instance based on type. """
        if store_type not in self.vector_store_configs:
            raise ValueError(f"Unsupported vector store type: {store_type}")

        config = self.vector_store_configs.get(store_type)
        store_class = config['class']

        vector_store = store_class(*args, **kwargs)
        return vector_store
