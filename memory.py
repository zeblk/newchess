import torch
import torch.nn as nn
from config import Config

# =============================================================================
# 1. THE SOUP (MEMORY WORKSPACE)
# =============================================================================
class MemoryWorkspace(nn.Module):
    """
    The Global Workspace. A single container for ALL types of information.
    There is no structural difference between 'Sensory Input' and 'Long Term Memory'.
    Everything is a vector with a 'Permanency' level and a 'Confidence' score.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Pre-allocate tensors for efficiency.
        # 'memories': The vector content (Concept Embeddings).
        # 'active_mask': Boolean flag for valid data vs empty slots.
        # 'permanency': The lifespan of a chunk. 
        #   0 = Ephemeral (cleared next step)
        #   >0 = Persists
        # 'confidence': How much we trust this chunk's predictions.
        self.register_buffer('memories', torch.zeros(config.MAX_MEMORIES, config.MEMORY_DIM))
        self.register_buffer('active_mask', torch.zeros(config.MAX_MEMORIES).bool())
        self.register_buffer('permanency', torch.zeros(config.MAX_MEMORIES))
        self.register_buffer('confidence', torch.zeros(config.MAX_MEMORIES))
        
        # Simple circular buffer pointer
        self.write_ptr = 0

    def add_chunks(self, chunks, initial_permanency=None, initial_confidence=None):
        """
        Injects new vectors into the soup.
        Used for incoming sensory data and for newly synthesized concepts.
        """
        if initial_permanency is None:
            initial_permanency = self.config.INITIAL_PERMANENCY_SENSORY
        if initial_confidence is None:
            initial_confidence = self.config.INITIAL_CONFIDENCE_SENSORY

        num_new = chunks.size(0)
        for i in range(num_new):
            idx = self.write_ptr
            self.memories[idx] = chunks[i]
            self.active_mask[idx] = True
            self.permanency[idx] = initial_permanency
            self.confidence[idx] = initial_confidence
            # Overwrite oldest (Circular Buffer)
            # Future Improvement: Overwrite lowest permanency (least useful) items instead.
            self.write_ptr = (self.write_ptr + 1) % self.config.MAX_MEMORIES

    def cleanup(self):
        """
        Removes ephemeral memories (permanency <= 0) at the start of a cycle.
        """
        # Identify items that should fade
        # For now, we just clear anything with <= 0 permanency.
        # In a more complex version, we might decay permanency here.
        to_clear = self.permanency <= 0
        self.active_mask[to_clear] = False
        self.permanency[to_clear] = 0.0 # Reset for safety
        self.confidence[to_clear] = 0.0

    def get_random_subset(self, k):
        """
        The 'Sampler'.
        Currently uniform random. In the future, this will be biased by Attention/Salience.
        Returns: (k, Memory_Dim) tensor of chunks, AND their indices in the workspace.
        """
        active_indices = torch.nonzero(self.active_mask).squeeze(1)
        if len(active_indices) < k:
            # Fallback for initialization phase
            # Return random noise and dummy indices (-1)
            return torch.randn(k, self.config.MEMORY_DIM), torch.full((k,), -1, dtype=torch.long)
        
        # Randomly select indices
        perm = torch.randperm(len(active_indices))[:k]
        selected_idx = active_indices[perm]
        return self.memories[selected_idx], selected_idx
        
    def get_top_k(self, k):
        """
        Returns the most 'active' chunks to drive motor output.
        We use permanency as a proxy for relevance here, or we could assume
        that if it's in the workspace, it's relevant.
        For now, let's just return the most permanent items, or random active ones.
        Actually, 'Energy' was activity. 'Permanency' is lifespan.
        Let's assume recently added or high permanency items are relevant.
        """
        masked_permanency = self.permanency.clone()
        masked_permanency[~self.active_mask] = -float('inf')
        
        # Safety check for empty memory
        k = min(k, self.active_mask.sum().item())
        if k == 0: return torch.zeros(self.config.TOP_K_FOR_ACTION, self.config.MEMORY_DIM)
        
        _, top_indices = torch.topk(masked_permanency, k)
        return self.memories[top_indices]

    def suppress(self, indices):
        """
        The 'Demotion' Mechanism.
        Called when a chunk is successfully explained by a simpler concept.
        We reduce its permanency to zero, effectively removing it from consideration.
        """
        self.permanency[indices] = 0.0

    def reinforce(self, chunk):
        """
        The 'Promotion' Mechanism.
        Called when a newly synthesized hypothesis successfully explains existing data.
        We commit it to memory with POSITIVE permanency so it persists.
        """
        self.add_chunks(chunk.unsqueeze(0), 
                        initial_permanency=self.config.INITIAL_PERMANENCY_CONCEPT,
                        initial_confidence=self.config.INITIAL_CONFIDENCE_CONCEPT)
