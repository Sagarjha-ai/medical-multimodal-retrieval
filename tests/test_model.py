"""
Test suite for model components
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('../app')

from models.multimodal_model import MultimodalModel

class TestMultimodalModel(unittest.TestCase):
    """Test multimodal model functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = MultimodalModel()
        self.batch_size = 2
        self.image_size = 224
        self.seq_length = 128
        
        # Create dummy inputs
        self.images = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        self.input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones(self.batch_size, self.seq_length)
    
    def test_model_initialization(self):
        """Test model initialization"""
        # Check that model is initialized
        self.assertIsInstance(self.model, nn.Module)
        
        # Check that components exist
        self.assertTrue(hasattr(self.model, 'image_encoder'))
        self.assertTrue(hasattr(self.model, 'text_encoder'))
        self.assertTrue(hasattr(self.model, 'image_projection'))
        self.assertTrue(hasattr(self.model, 'text_projection'))
    
    def test_forward_pass(self):
        """Test forward pass"""
        # Set model to eval mode
        self.model.eval()
        
        with torch.no_grad():
            image_embeddings, text_embeddings = self.model(
                self.images,
                self.input_ids,
                self.attention_mask
            )
        
        # Check output shapes
        expected_shape = (self.batch_size, 512)  # Embedding dimension
        self.assertEqual(image_embeddings.shape, expected_shape)
        self.assertEqual(text_embeddings.shape, expected_shape)
        
        # Check that embeddings are normalized
        image_norms = torch.norm(image_embeddings, dim=-1)
        text_norms = torch.norm(text_embeddings, dim=-1)
        
        # Should be close to 1.0 due to L2 normalization
        self.assertTrue(torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-6))
        self.assertTrue(torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-6))
    
    def test_embedding_similarity(self):
        """Test embedding similarity computation"""
        self.model.eval()
        
        with torch.no_grad():
            image_embeddings, text_embeddings = self.model(
                self.images,
                self.input_ids,
                self.attention_mask
            )
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)
        
        # Check similarity matrix shape
        self.assertEqual(similarity_matrix.shape, (self.batch_size, self.batch_size))
        
        # Check that diagonal elements (matching pairs) have higher similarity
        diagonal_similarities = torch.diag(similarity_matrix)
        off_diagonal_similarities = similarity_matrix[~torch.eye(self.batch_size, dtype=bool)]
        
        # On average, diagonal should be higher than off-diagonal
        self.assertGreater(diagonal_similarities.mean(), off_diagonal_similarities.mean())
    
    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        self.model.train()
        
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Forward pass
        image_embeddings, text_embeddings = self.model(
            self.images,
            self.input_ids,
            self.attention_mask
        )
        
        # Compute a simple loss
        loss = torch.sum(image_embeddings * text_embeddings)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        self.assertTrue(has_gradients, "No gradients computed during backward pass")
    
    def test_parameter_freezing(self):
        """Test parameter freezing functionality"""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze projection heads
        for param in self.model.image_projection.parameters():
            param.requires_grad = True
        for param in self.model.text_projection.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Should only have projection head parameters trainable
        projection_params = (
            sum(p.numel() for p in self.model.image_projection.parameters()) +
            sum(p.numel() for p in self.model.text_projection.parameters())
        )
        
        self.assertEqual(trainable_params, projection_params)
        self.assertLess(trainable_params, total_params)

class TestContrastiveLoss(unittest.TestCase):
    """Test contrastive loss function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.embedding_dim = 512
        self.temperature = 0.07
        
        # Create normalized embeddings
        self.image_embeddings = torch.randn(self.batch_size, self.embedding_dim)
        self.image_embeddings = torch.nn.functional.normalize(self.image_embeddings, dim=-1)
        
        self.text_embeddings = torch.randn(self.batch_size, self.embedding_dim)
        self.text_embeddings = torch.nn.functional.normalize(self.text_embeddings, dim=-1)
    
    def test_contrastive_loss_computation(self):
        """Test contrastive loss computation"""
        # Define contrastive loss
        class ContrastiveLoss(nn.Module):
            def __init__(self, temperature=0.07):
                super().__init__()
                self.temperature = temperature
            
            def forward(self, image_embeddings, text_embeddings):
                logits = (image_embeddings @ text_embeddings.T) / self.temperature
                labels = torch.arange(logits.shape[0])
                
                loss_i = nn.functional.cross_entropy(logits, labels)
                loss_t = nn.functional.cross_entropy(logits.T, labels)
                
                return (loss_i + loss_t) / 2
        
        criterion = ContrastiveLoss(self.temperature)
        loss = criterion(self.image_embeddings, self.text_embeddings)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0)
    
    def test_loss_symmetry(self):
        """Test that loss is symmetric"""
        class ContrastiveLoss(nn.Module):
            def __init__(self, temperature=0.07):
                super().__init__()
                self.temperature = temperature
            
            def forward(self, image_embeddings, text_embeddings):
                logits = (image_embeddings @ text_embeddings.T) / self.temperature
                labels = torch.arange(logits.shape[0])
                
                loss_i = nn.functional.cross_entropy(logits, labels)
                loss_t = nn.functional.cross_entropy(logits.T, labels)
                
                return (loss_i + loss_t) / 2
        
        criterion = ContrastiveLoss(self.temperature)
        loss1 = criterion(self.image_embeddings, self.text_embeddings)
        loss2 = criterion(self.text_embeddings, self.image_embeddings)
        
        # Loss should be symmetric
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6)

if __name__ == '__main__':
    unittest.main()
