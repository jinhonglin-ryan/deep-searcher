import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from deepsearcher.vector_db import Milvus
from deepsearcher.loader.splitter import Chunk
from deepsearcher.vector_db.base import RetrievalResult


class TestMilvus(unittest.TestCase):
    """Simple tests for the Milvus vector database implementation."""

    @patch('pymilvus.connections')
    @patch('pymilvus.MilvusClient')
    def test_init(self, mock_client_class, mock_connections):
        """Test basic initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # Mock the connections.connect to prevent actual connection attempts
        mock_connections.connect = MagicMock()
        
        milvus = Milvus(
            default_collection="test_collection",
            uri="http://localhost:19530",
            token="root:Milvus",
            db="default",
            hybrid=False
        )
        
        # Verify initialization - just check basic properties
        self.assertEqual(milvus.default_collection, "test_collection")
        self.assertFalse(milvus.hybrid)
        self.assertIsNotNone(milvus.client)

    @patch('pymilvus.connections')
    @patch('pymilvus.MilvusClient')
    def test_init_collection(self, mock_client_class, mock_connections):
        """Test collection initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # Mock the connections.connect to prevent actual connection attempts
        mock_connections.connect = MagicMock()
        mock_client.has_collection.return_value = False  # Collection doesn't exist
        
        # Mock schema and index creation
        mock_schema = MagicMock()
        mock_index_params = MagicMock()
        mock_client.create_schema.return_value = mock_schema
        mock_client.prepare_index_params.return_value = mock_index_params
        
        milvus = Milvus()
        
        # Test collection initialization
        d = 8
        collection = "hello_deepsearcher"
        
        try:
            milvus.init_collection(dim=d, collection=collection)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "init_collection should work")

    @patch('pymilvus.connections')
    @patch('pymilvus.MilvusClient')
    def test_insert_data_with_retrieval_results(self, mock_client_class, mock_connections):
        """Test inserting data using RetrievalResult objects."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # Mock the connections.connect to prevent actual connection attempts
        mock_connections.connect = MagicMock()
        mock_client.insert.return_value = None
        
        milvus = Milvus()
        
        # Create test data
        d = 8
        collection = "hello_deepsearcher"
        rng = np.random.default_rng(seed=19530)
        
        # Create RetrievalResult objects
        test_data = [
            RetrievalResult(
                embedding=rng.random((1, d))[0],
                text="hello world",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
            RetrievalResult(
                embedding=rng.random((1, d))[0],
                text="hello milvus",
                reference="local file: hi.txt",
                metadata={"a": 1},
            ),
        ]
        
        try:
            milvus.insert_data(collection=collection, chunks=test_data)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "insert_data should work with RetrievalResult objects")

    @patch('pymilvus.connections')
    @patch('pymilvus.MilvusClient')
    def test_search_data(self, mock_client_class, mock_connections):
        """Test search functionality."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # Mock the connections.connect to prevent actual connection attempts
        mock_connections.connect = MagicMock()
        
        # Mock search results
        d = 8
        rng = np.random.default_rng(seed=19530)
        mock_search_results = [[
            {
                "entity": {
                    "embedding": rng.random((1, d))[0].tolist(),
                    "text": "hello world",
                    "reference": "local file: hi.txt",
                    "metadata": {"a": 1}
                },
                "distance": 0.5
            },
            {
                "entity": {
                    "embedding": rng.random((1, d))[0].tolist(),
                    "text": "hello milvus",
                    "reference": "local file: hi.txt",
                    "metadata": {"a": 1}
                },
                "distance": 0.8
            }
        ]]
        mock_client.search.return_value = mock_search_results
        
        milvus = Milvus()
        
        # Test search
        collection = "hello_deepsearcher"
        query_vector = rng.random((1, d))[0]
        
        try:
            top_2 = milvus.search_data(
                collection=collection, 
                vector=query_vector, 
                top_k=2
            )
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "search_data should work")
        if test_passed:
            self.assertIsInstance(top_2, list)
            self.assertEqual(len(top_2), 2)
            # Verify results are RetrievalResult objects
            for result in top_2:
                self.assertIsInstance(result, RetrievalResult)

    @patch('pymilvus.connections')
    @patch('pymilvus.MilvusClient')
    def test_clear_collection(self, mock_client_class, mock_connections):
        """Test clearing collection."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # Mock the connections.connect to prevent actual connection attempts
        mock_connections.connect = MagicMock()
        mock_client.drop_collection.return_value = None
        
        milvus = Milvus()
        
        collection = "hello_deepsearcher"
        
        try:
            milvus.clear_db(collection=collection)
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "clear_db should work")

    @patch('pymilvus.connections')
    @patch('pymilvus.MilvusClient')
    def test_list_collections(self, mock_client_class, mock_connections):
        """Test listing collections."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        # Mock the connections.connect to prevent actual connection attempts
        mock_connections.connect = MagicMock()
        mock_client.list_collections.return_value = ["hello_deepsearcher", "test_collection"]
        
        # Mock describe_collection for each collection
        def mock_describe(collection_name):
            return {
                "description": f"Description for {collection_name}",
                "fields": []
            }
        mock_client.describe_collection.side_effect = mock_describe
        
        milvus = Milvus()
        
        try:
            collections = milvus.list_collections()
            test_passed = True
        except Exception as e:
            test_passed = False
            print(f"Error: {e}")
        
        self.assertTrue(test_passed, "list_collections should work")
        if test_passed:
            self.assertIsInstance(collections, list)
            self.assertGreaterEqual(len(collections), 0)


if __name__ == "__main__":
    unittest.main() 