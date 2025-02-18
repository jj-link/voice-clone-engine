import torch
from pynvml import *
from typing import List, Dict, Optional, Union
import logging
import re

class GPUManager:
    def __init__(self):
        """Initialize the GPU manager."""
        self.logger = logging.getLogger(__name__)
        self._is_shutdown = False
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
        self.devices = self._initialize_devices()

    def _initialize_devices(self) -> Dict[int, Dict]:
        """Initialize and gather information about all available GPUs."""
        devices = {}
        for i in range(self.device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            device = {
                'handle': handle,
                'name': nvmlDeviceGetName(handle),
                'total_memory': info.total,
                'compute_capability': torch.cuda.get_device_capability(i),
                'bus_id': nvmlDeviceGetPciInfo(handle).busId
            }
            devices[i] = device
        return devices

    def _check_shutdown(self):
        """Check if the manager has been shut down."""
        if self._is_shutdown:
            raise RuntimeError("GPU Manager has been shut down")

    def parse_gpu_args(self, gpu_str: str) -> List[int]:
        """Parse GPU selection argument string into list of GPU indices.
        
        Args:
            gpu_str: String specifying GPU selection (e.g., "all", "0,2,5", "range:0-3")
        
        Returns:
            List of GPU indices to use
            
        Raises:
            ValueError: If the gpu_str format is invalid or specifies non-existent GPUs
        """
        self._check_shutdown()
        if not isinstance(gpu_str, str):
            raise ValueError("GPU argument must be a string")

        if gpu_str.lower() == 'all':
            return list(range(self.device_count))
        
        if gpu_str.startswith('range:'):
            try:
                start, end = map(int, gpu_str[6:].split('-'))
                if start < 0 or end >= self.device_count or start > end:
                    raise ValueError(
                        f"Invalid range {start}-{end}. Must be between 0 and {self.device_count-1}"
                    )
                return list(range(start, end + 1))
            except ValueError as e:
                if "map" in str(e):
                    raise ValueError("Range format must be 'range:start-end'")
                raise
        
        if gpu_str.startswith('count:'):
            try:
                count = int(gpu_str[6:])
                if count <= 0 or count > self.device_count:
                    raise ValueError(
                        f"Invalid GPU count {count}. Must be between 1 and {self.device_count}"
                    )
                return self._select_best_gpus(count)
            except ValueError:
                raise ValueError("Count format must be 'count:n' where n is an integer")

        # Handle comma-separated list
        try:
            indices = [int(idx) for idx in gpu_str.split(',')]
            invalid_indices = [idx for idx in indices if idx >= self.device_count or idx < 0]
            if invalid_indices:
                raise ValueError(
                    f"Invalid GPU indices: {invalid_indices}. "
                    f"Must be between 0 and {self.device_count-1}"
                )
            return indices
        except ValueError:
            raise ValueError(
                "GPU indices must be comma-separated integers or 'all', 'range:start-end', 'count:n'"
            )

    def _select_best_gpus(self, count: int) -> List[int]:
        """Select the best GPUs based on memory availability and compute capability."""
        self._check_shutdown()
        if count > self.device_count:
            raise ValueError(f"Requested {count} GPUs but only {self.device_count} available")
            
        gpu_metrics = []
        for idx, device in self.devices.items():
            info = nvmlDeviceGetMemoryInfo(device['handle'])
            free_memory = info.free
            compute_score = device['compute_capability'][0] * 100 + device['compute_capability'][1]
            gpu_metrics.append((idx, free_memory, compute_score))
        
        # Sort by free memory and compute capability
        gpu_metrics.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [idx for idx, _, _ in gpu_metrics[:count]]

    def get_gpu_info(self, gpu_indices: Optional[List[int]] = None) -> Dict:
        """Get current information about specified GPUs."""
        self._check_shutdown()
        
        if gpu_indices is None:
            gpu_indices = list(range(self.device_count))
            
        # Validate indices
        invalid_indices = [idx for idx in gpu_indices if idx >= self.device_count or idx < 0]
        if invalid_indices:
            raise ValueError(
                f"Invalid GPU indices: {invalid_indices}. "
                f"Must be between 0 and {self.device_count-1}"
            )
        
        info = {}
        for idx in gpu_indices:
            handle = self.devices[idx]['handle']
            memory = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            
            info[idx] = {
                'name': self.devices[idx]['name'],
                'free_memory': memory.free,
                'total_memory': memory.total,
                'used_memory': memory.used,
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory,
                'compute_capability': self.devices[idx]['compute_capability']
            }
        return info

    def get_gpu_with_most_memory(self) -> int:
        """Get the GPU with the most free memory."""
        self._check_shutdown()
        gpu_metrics = []
        for idx, device in self.devices.items():
            info = nvmlDeviceGetMemoryInfo(device['handle'])
            gpu_metrics.append((idx, info.free))
        gpu_metrics.sort(key=lambda x: x[1], reverse=True)
        return gpu_metrics[0][0]

    def get_device(self) -> torch.device:
        """Get the most suitable device for processing.
        
        Returns:
            torch.device: The selected device (CPU or GPU)
        """
        if not self.device_count:
            return torch.device('cpu')
            
        # Get the GPU with the most free memory
        gpu_id = self.get_gpu_with_most_memory()
        return torch.device(f'cuda:{gpu_id}')

    def setup_torch_devices(self, gpu_indices: List[int], memory_limit: Optional[int] = None):
        """Setup PyTorch to use specified GPUs with optional memory limits.
        
        Args:
            gpu_indices: List of GPU indices to use
            memory_limit: Optional memory limit as percentage (0-100)
            
        Returns:
            List of torch.device objects for specified GPUs, or single CPU device if no GPUs
        """
        self._check_shutdown()
        if not gpu_indices:
            self.logger.warning("No GPUs specified, using CPU")
            return torch.device('cpu')

        # Validate memory limit
        if memory_limit is not None:
            if not 0 < memory_limit <= 100:
                raise ValueError("Memory limit must be between 1 and 100")
            
            for idx in gpu_indices:
                total_memory = self.devices[idx]['total_memory']
                limit_fraction = memory_limit / 100
                torch.cuda.set_per_process_memory_fraction(limit_fraction, idx)

        # Set visible devices
        visible_devices = ','.join(map(str, gpu_indices))
        torch.cuda.set_device(gpu_indices[0])  # Set primary GPU
        return [torch.device(f'cuda:{idx}') for idx in gpu_indices]

    def cleanup(self):
        """Cleanup NVML initialization."""
        if not self._is_shutdown:
            try:
                nvmlShutdown()
                self._is_shutdown = True
            except Exception as e:
                self.logger.warning(f"Error during NVML shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
