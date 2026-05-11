"""
Test suite for data loading functionality
"""

import unittest
import torch
import pandas as pd
import numpy as np
from PIL import Image
import tempfile
import os

import sys
sys.path.append('../app')

from data.data_loader import ChestXrayDataset, apply_clahe, get_train_transforms, get_val_transforms

class TestDataLoader(unittest.TestCase):
    """Test data loading and preprocessing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'image_path': ['test1.jpg', 'test2.jpg'],
            'structured_caption': [
                'Normal chest X-ray with no acute findings',
                'Pneumonia with right lower lobe opacity'
            ]
        })
        
        # Create temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy images
        for i, img_path in enumerate(self.sample_data['image_path']):
            full_path = os.path.join(self.temp_dir, img_path)
            # Create a simple grayscale image
            img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L').convert('RGB')
            img.save(full_path)
        
        # Update paths in dataframe
        self.sample_data['image_path'] = [
            os.path.join(self.temp_dir, img_path) 
            for img_path in self.sample_data['image_path']
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_apply_clahe(self):
        """Test CLAHE enhancement function"""
        # Create test image
        test_img = Image.new('RGB', (224, 224), color='gray')
        
        # Apply CLAHE
        enhanced_img = apply_clahe(test_img)
        
        # Check that output is still a PIL Image
        self.assertIsInstance(enhanced_img, Image.Image)
        
        # Check that dimensions are preserved
        self.assertEqual(enhanced_img.size, (224, 224))
        
        # Check that mode is still RGB
        self.assertEqual(enhanced_img.mode, 'RGB')
    
    def test_dataset_initialization(self):
        """Test dataset initialization"""
        transforms = get_train_transforms()
        dataset = ChestXrayDataset(self.sample_data, transforms)
        
        # Check dataset length
        self.assertEqual(len(dataset), 2)
        
        # Check that data is stored correctly
        self.assertEqual(len(dataset.dataframe), 2)
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        transforms = get_val_transforms()
        dataset = ChestXrayDataset(self.sample_data, transforms)
        
        # Get first item
        item = dataset[0]
        
        # Check item structure
        self.assertIn('image', item)
        self.assertIn('caption', item)
        self.assertIn('image_path', item)
        
        # Check image tensor
        self.assertIsInstance(item['image'], torch.Tensor)
        self.assertEqual(item['image'].shape, (3, 224, 224))
        
        # Check caption
        self.assertIsInstance(item['caption'], str)
        self.assertEqual(item['caption'], self.sample_data.iloc[0]['structured_caption'])
    
    def test_transforms(self):
        """Test transform functions"""
        # Test train transforms
        train_transforms = get_train_transforms(224)
        test_img = Image.new('RGB', (224, 224), color='gray')
        transformed = train_transforms(test_img)
        
        self.assertIsInstance(transformed, torch.Tensor)
        self.assertEqual(transformed.shape, (3, 224, 224))
        
        # Test validation transforms
        val_transforms = get_val_transforms(224)
        transformed_val = val_transforms(test_img)
        
        self.assertIsInstance(transformed_val, torch.Tensor)
        self.assertEqual(transformed_val.shape, (3, 224, 224))
    
    def test_dataloader_compatibility(self):
        """Test dataset compatibility with DataLoader"""
        from torch.utils.data import DataLoader
        
        transforms = get_val_transforms()
        dataset = ChestXrayDataset(self.sample_data, transforms)
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get batch
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('image', batch)
        self.assertIn('caption', batch)
        self.assertIn('image_path', batch)
        
        # Check batch shapes
        self.assertEqual(batch['image'].shape, (2, 3, 224, 224))
        self.assertEqual(len(batch['caption']), 2)
        self.assertEqual(len(batch['image_path']), 2)

class TestModelIntegration(unittest.TestCase):
    """Test integration with model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'image_path': ['test.jpg'],
            'structured_caption': ['Test caption for medical image'],
            'input_ids': [[101, 102, 103, 104, 102] * 25 + [102]],  # Dummy token IDs
            'attention_mask': [[1] * 128]  # Dummy attention mask
        })
    
    def test_tokenization_compatibility(self):
        """Test that captions can be tokenized properly"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        for caption in self.sample_data['structured_caption']:
            # Test tokenization
            tokens = tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Check token structure
            self.assertIn('input_ids', tokens)
            self.assertIn('attention_mask', tokens)
            
            # Check tensor shapes
            self.assertEqual(tokens['input_ids'].shape, (1, 128))
            self.assertEqual(tokens['attention_mask'].shape, (1, 128))

if __name__ == '__main__':
    unittest.main()
