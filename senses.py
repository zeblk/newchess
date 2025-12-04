import torch
import torch.nn as nn
from config import Config

# =============================================================================
# 2. SENSORY CORTEX
# =============================================================================
class SensoryCortex(nn.Module):
    """
    Processes raw inputs (pixels) into compatible Memory Chunks.
    Does NOT classify. Just extracts features and projects them to memory space.
    Also returns the 'Cortical State' (activations of all layers) for consistency checking.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Split conv layers to access intermediate activations
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.INPUT_CHANNELS, config.CONV1_CHANNELS, kernel_size=config.CONV_KERNEL_SIZE, stride=config.CONV_STRIDE, padding=config.CONV_PADDING),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(config.CONV1_CHANNELS, config.CONV2_CHANNELS, kernel_size=config.CONV_KERNEL_SIZE, stride=config.CONV_STRIDE, padding=config.CONV_PADDING),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        
        # We assume the CNN output can be sliced into multiple distinct chunks.
        self.projection = nn.Linear(config.FLATTENED_FEATURE_SIZE, config.MEMORY_DIM * config.SENSORY_CHUNKS)

    def forward(self, x):
        # 1. Forward Pass with Capture
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        flat = self.flatten(c2)
        
        # 2. Create Chunks
        # Reshape to create discrete objects/chunks from the visual field
        chunks = self.projection(flat).view(-1, self.config.SENSORY_CHUNKS, self.config.MEMORY_DIM) 
        
        # 3. Create Cortical State
        # Concatenate flattened activations of all layers
        # c1: (B, 16, 8, 8) -> (B, 1024)
        # c2: (B, 32, 8, 8) -> (B, 2048)
        cortical_state = torch.cat([c1.flatten(start_dim=1), c2.flatten(start_dim=1)], dim=1)
        
        return chunks.squeeze(0), cortical_state.squeeze(0)
