import torch
import torch.nn as nn
from config import Config

# =============================================================================
# 4. MOTOR CORTEX
# =============================================================================
class MotorCortex(nn.Module):
    """
    Translates 'State of Mind' into Action.
    Reads the most energetic memories and computes policy logits.
    """
    def __init__(self, config):
        super().__init__()
        input_dim = config.TOP_K_FOR_ACTION * config.MEMORY_DIM
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, config.MOTOR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.MOTOR_HIDDEN_DIM, config.ACTION_SPACE)
        )

    def forward(self, top_memories):
        # Flatten the set of top memories into a single state vector
        flat = top_memories.view(-1)
        
        # Handle edge case where we don't have enough memories yet (padding)
        target_size = self.policy_net[0].in_features
        if flat.size(0) < target_size:
            padding = torch.zeros(target_size - flat.size(0))
            flat = torch.cat([flat, padding])
            
        return self.policy_net(flat)
