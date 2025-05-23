import unittest
import os
from unittest.mock import patch, MagicMock

import requests
from deepsearcher.embedding import SiliconflowEmbedding


class TestSiliconflowEmbedding(unittest.TestCase):
    """Tests for the SiliconflowEmbedding class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patches for requests
        self.requests_patcher = patch('requests.request')
        self.mock_request = self.requests_patcher.start()
        
        # Set up mock response
        self.mock_response = MagicMock()
        self.mock_response.json.return_value = {
            'data': [
                {'index': 0, 'embedding': [0.1] * 1024}  # BAAI/bge-m3 has 1024 dimensions
            ]
        }
        self.mock_response.raise_for_status = MagicMock()
        self.mock_request.return_value = self.mock_response
        
        # Set environment variable for API key
        self.env_patcher = patch.dict('os.environ', {'SILICONFLOW_API_KEY': 'fake-api-key'})
        self.env_patcher.start()
        
        # Create the embedder
        self.embedding = SiliconflowEmbedding()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.requests_patcher.stop()
        self.env_patcher.stop()
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        # Check attributes
        self.assertEqual(self.embedding.model, 'BAAI/bge-m3')
        self.assertEqual(self.embedding.api_key, 'fake-api-key')
        self.assertEqual(self.embedding.batch_size, 32)
    
    def test_init_with_model(self):
        """Test initialization with specified model."""
        # Initialize with a different model
        embedding = SiliconflowEmbedding(model='netease-youdao/bce-embedding-base_v1')
        
        # Check attributes
        self.assertEqual(embedding.model, 'netease-youdao/bce-embedding-base_v1')
        self.assertEqual(embedding.dimension, 768)
    
    def test_init_with_model_name(self):
        """Test initialization with model_name parameter."""
        # Initialize with model_name
        embedding = SiliconflowEmbedding(model_name='BAAI/bge-large-zh-v1.5')
        
        # Check attributes
        self.assertEqual(embedding.model, 'BAAI/bge-large-zh-v1.5')
    
    def test_init_with_api_key(self):
        """Test initialization with API key parameter."""
        # Initialize with API key
        embedding = SiliconflowEmbedding(api_key='test-api-key')
        
        # Check that the API key was set correctly
        self.assertEqual(embedding.api_key, 'test-api-key')
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        # Remove API key from environment
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError):
                SiliconflowEmbedding()
    
    def test_embed_query(self):
        """Test embedding a single query."""
        # Create a test query
        query = "This is a test query"
        
        # Call the method
        result = self.embedding.embed_query(query)
        
        # Verify that request was called correctly
        self.mock_request.assert_called_once_with(
            'POST',
            'https://api.siliconflow.cn/v1/embeddings',
            json={
                'model': 'BAAI/bge-m3',
                'input': query,
                'encoding_format': 'float'
            },
            headers={
                'Authorization': 'Bearer fake-api-key',
                'Content-Type': 'application/json'
            }
        )
        
        # Check the result
        self.assertEqual(result, [0.1] * 1024)
    
    def test_embed_documents(self):
        """Test embedding multiple documents."""
        # Create test documents
        texts = ["text 1", "text 2", "text 3"]
        
        # Set up mock response for multiple documents
        self.mock_response.json.return_value = {
            'data': [
                {'index': i, 'embedding': [0.1 * (i + 1)] * 1024}
                for i in range(3)
            ]
        }
        
        # Call the method
        results = self.embedding.embed_documents(texts)
        
        # Verify that request was called correctly
        self.mock_request.assert_called_once_with(
            'POST',
            'https://api.siliconflow.cn/v1/embeddings',
            json={
                'model': 'BAAI/bge-m3',
                'input': texts,
                'encoding_format': 'float'
            },
            headers={
                'Authorization': 'Bearer fake-api-key',
                'Content-Type': 'application/json'
            }
        )
        
        # Check the results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result, [0.1 * (i + 1)] * 1024)
    
    def test_embed_documents_with_batching(self):
        """Test embedding documents with batching."""
        # Create test documents
        texts = ["text " + str(i) for i in range(50)]  # More than batch_size
        
        # Set up mock response for batched documents
        def mock_batch_response(*args, **kwargs):
            batch_input = kwargs['json']['input']
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                'data': [
                    {'index': i, 'embedding': [0.1] * 1024}
                    for i in range(len(batch_input))
                ]
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp
        
        self.mock_request.side_effect = mock_batch_response
        
        # Call the method
        results = self.embedding.embed_documents(texts)
        
        # Check that request was called multiple times
        self.assertTrue(self.mock_request.call_count > 1)
        
        # Check the results
        self.assertEqual(len(results), 50)
        for result in results:
            self.assertEqual(result, [0.1] * 1024)
    
    def test_dimension_property(self):
        """Test the dimension property."""
        # For BAAI/bge-m3
        self.assertEqual(self.embedding.dimension, 1024)
        
        # For netease-youdao/bce-embedding-base_v1
        embedding = SiliconflowEmbedding(model='netease-youdao/bce-embedding-base_v1')
        self.assertEqual(embedding.dimension, 768)
        
        # For BAAI/bge-large-zh-v1.5
        embedding = SiliconflowEmbedding(model='BAAI/bge-large-zh-v1.5')
        self.assertEqual(embedding.dimension, 1024) 

if __name__ == "__main__":
    unittest.main() 