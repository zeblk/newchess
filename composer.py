import torch
import torch.nn as nn
from config import Config

# =============================================================================
# 3. CONCEPT COMPOSER (TEMPLATE-DRIVEN ATTENTION)
# =============================================================================
class ConceptComposer(nn.Module):
    """
    The 'Hypothesis Generator'.
    Solves the Variable Binding Problem using a Template-Key-Value mechanism.
    
    1. Template -> Roles (Queries)
    2. Arguments -> Fillers (Keys/Values)
    3. Roles attend to Fillers -> Bound Slots
    4. Fusion -> New Concept
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.MEMORY_DIM
        self.num_slots = config.INTERNAL_SLOTS

        # A. Query Generator (Structure)
        # Projects a Template Vector (e.g., "Cooking") into distinct Role Queries
        # (e.g., [Agent_Query, Location_Query, Ingredient_Query])
        self.query_generator = nn.Linear(self.dim, self.dim * self.num_slots)

        # B. Binder (Syntax)
        # Standard Multihead Attention. 
        # The Roles (Queries) scan the Arguments (Keys) to find the best fit.
        self.binder = nn.MultiheadAttention(embed_dim=self.dim, num_heads=config.ATTENTION_HEADS, batch_first=True)

        # C. Fusion (Compression)
        # Compresses the bound structure back into a single vector.
        self.f_phi = nn.Sequential(
            nn.Linear(self.dim, config.CONCEPT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.CONCEPT_HIDDEN_DIM, self.dim)
        )

        # D. Predictor (Causality Check)
        # Projects the abstract Concept (Cause) to Expected Evidence (Effect).
        # Essential because the Concept Vector != The Evidence Vector.
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, config.CONCEPT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.CONCEPT_HIDDEN_DIM, config.CORTICAL_DIM)
        )

    def forward(self, template, arguments):
        """
        Inputs:
            template: (Batch, Dim) - The Chunk acting as the 'Schema'.
            arguments: (Batch, Num_Args, Dim) - The Chunks acting as 'Fillers'.
        Returns:
            m_new: The synthesized Concept vector.
            predicted_evidence: What this concept implies we should see.
        """
        b = template.size(0)

        # 1. GENERATE ROLES
        # "Cooking" becomes -> [Looking for Cook, Looking for Kitchen, ...]
        queries = self.query_generator(template).view(b, self.num_slots, self.dim)

        # 2. BIND
        # The Queries attend to the Arguments.
        # Result: 'bound_slots' contains the Arguments weighted by how well they fit the Roles.
        bound_slots, attn_weights = self.binder(query=queries, key=arguments, value=arguments)

        # 3. FUSE
        # We sum the original Template (to retain the "Schema" identity)
        # with the Bound Slots (the "Fillers").
        # This Sum is a Superposition of the whole thought.
        superposition = template + bound_slots.sum(dim=1)
        
        # Project back to the canonical Memory Dimension.
        m_new = self.f_phi(superposition)
        
        # 4. PREDICT CONSEQUENCES
        # Predict sensory implications of this concept.
        predicted_evidence = self.predictor(m_new)
        
        return m_new, predicted_evidence
