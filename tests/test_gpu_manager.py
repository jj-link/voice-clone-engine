import unittest
import sys
import os
import torch
import yaml
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gpu_manager import GPUManager

class TestGPUManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.gpu_manager = GPUManager()
        
        # Create a mock config for testing
        self.test_config = {
            'gpu_settings': {
                'selection': 'all',
                'memory_limit': 90,
                'strategy': 'balanced',
                'monitoring': {
                    'enabled': True,
                    'interval_seconds': 5
                }
            }
        }

    def tearDown(self):
        """Clean up after each test method."""
        self.gpu_manager.cleanup()

    def test_gpu_detection(self):
        """Test that GPUs are properly detected."""
        self.assertGreaterEqual(self.gpu_manager.device_count, 0)
        self.assertIsInstance(self.gpu_manager.devices, dict)
        
        # Test device information structure
        if self.gpu_manager.device_count > 0:
            first_device = self.gpu_manager.devices[0]
            self.assertIn('name', first_device)
            self.assertIn('total_memory', first_device)
            self.assertIn('compute_capability', first_device)
            self.assertIn('bus_id', first_device)

    def test_parse_gpu_args(self):
        """Test all GPU argument parsing scenarios."""
        # Test 'all' keyword
        all_gpus = self.gpu_manager.parse_gpu_args('all')
        self.assertEqual(len(all_gpus), self.gpu_manager.device_count)
        
        # Test specific indices
        if self.gpu_manager.device_count >= 2:
            specific_gpus = self.gpu_manager.parse_gpu_args('0,1')
            self.assertEqual(specific_gpus, [0, 1])
            
            # Test single GPU
            single_gpu = self.gpu_manager.parse_gpu_args('0')
            self.assertEqual(single_gpu, [0])
        
        # Test range format
        if self.gpu_manager.device_count >= 3:
            range_gpus = self.gpu_manager.parse_gpu_args('range:0-2')
            self.assertEqual(range_gpus, [0, 1, 2])
        
        # Test count format
        if self.gpu_manager.device_count >= 2:
            count_gpus = self.gpu_manager.parse_gpu_args('count:2')
            self.assertEqual(len(count_gpus), 2)
            self.assertTrue(all(isinstance(idx, int) for idx in count_gpus))

    def test_invalid_gpu_args(self):
        """Test handling of invalid GPU arguments."""
        with self.assertRaises(ValueError):
            self.gpu_manager.parse_gpu_args('invalid')
        
        with self.assertRaises(ValueError):
            self.gpu_manager.parse_gpu_args('range:invalid')
        
        with self.assertRaises(ValueError):
            self.gpu_manager.parse_gpu_args(f'{self.gpu_manager.device_count + 1}')

    def test_gpu_info(self):
        """Test GPU information retrieval."""
        if self.gpu_manager.device_count > 0:
            info = self.gpu_manager.get_gpu_info([0])
            
            # Test structure of GPU info
            self.assertIn(0, info)
            gpu_info = info[0]
            required_keys = [
                'name', 'free_memory', 'total_memory', 'used_memory',
                'gpu_utilization', 'memory_utilization', 'compute_capability'
            ]
            for key in required_keys:
                self.assertIn(key, gpu_info)
                
            # Test memory values make sense
            self.assertGreater(gpu_info['total_memory'], 0)
            self.assertEqual(
                gpu_info['total_memory'],
                gpu_info['free_memory'] + gpu_info['used_memory']
            )

    def test_select_best_gpus(self):
        """Test the GPU selection algorithm."""
        if self.gpu_manager.device_count >= 2:
            # Test selecting best GPU
            best_gpus = self.gpu_manager._select_best_gpus(1)
            self.assertEqual(len(best_gpus), 1)
            
            # Test selecting multiple GPUs
            multi_gpus = self.gpu_manager._select_best_gpus(2)
            self.assertEqual(len(multi_gpus), 2)
            self.assertEqual(len(set(multi_gpus)), 2)  # Check for uniqueness

    @patch('torch.cuda')
    def test_setup_torch_devices(self, mock_cuda):
        """Test PyTorch device setup with different configurations."""
        # Mock cuda device count
        mock_cuda.device_count.return_value = 2
        
        # Test setup with specific GPUs
        devices = self.gpu_manager.setup_torch_devices([0, 1], memory_limit=80)
        self.assertEqual(len(devices), 2)
        self.assertTrue(all(isinstance(d, torch.device) for d in devices))
        
        # Test setup with memory limit
        devices = self.gpu_manager.setup_torch_devices([0], memory_limit=50)
        mock_cuda.set_per_process_memory_fraction.assert_called_with(0.5, 0)
        
        # Test setup with no GPUs
        cpu_device = self.gpu_manager.setup_torch_devices([])
        self.assertEqual(cpu_device, torch.device('cpu'))

    @patch('src.gpu_manager.nvmlShutdown')
    def test_context_manager(self, mock_shutdown):
        """Test that the GPU manager works as a context manager."""
        with GPUManager() as manager:
            self.assertIsInstance(manager, GPUManager)
            # Perform some operation
            info = manager.get_gpu_info()
            self.assertIsInstance(info, dict)
        
        # Verify shutdown was called
        mock_shutdown.assert_called_once()
        
        # After context manager exits, operations should fail
        with self.assertRaises(Exception):
            manager.get_gpu_info()

if __name__ == '__main__':
    unittest.main()
