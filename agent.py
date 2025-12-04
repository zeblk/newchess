import torch
import torch.nn as nn
from config import Config
from memory import MemoryWorkspace
from senses import SensoryCortex
from composer import ConceptComposer
from motor import MotorCortex

# =============================================================================
# THE RADICAL ARCHITECTURE
# =============================================================================
# A Generative, Abductive Agent that "thinks" by sampling combinations of
# memories to find simpler explanations for its experiences.
#
# Core Loop:
# 1. GENERATE: Sample a Template + Arguments and bind them into a Hypothesis.
# 2. BROADCAST: Project the Hypothesis into "Evidence Space".
# 3. RESOLVE: If Evidence matches existing Memory Chunks, suppress them (explain)
#             and reinforce the Hypothesis (learn).
# =============================================================================

# =============================================================================
# 5. THE RADICAL AGENT (MAIN SYSTEM)
# =============================================================================
class RadicalAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        
        # The Modules
        self.memory = MemoryWorkspace(self.config)
        self.senses = SensoryCortex(self.config)
        self.composer = ConceptComposer(self.config)
        self.motor = MotorCortex(self.config)

    def observe(self, sensory_input):
        """
        Phase 1: SENSE
        Processes sensory input and injects it into the memory workspace.
        """
        # 0. Cleanup Ephemeral Memories
        self.memory.cleanup()
        
        # Convert raw input to chunks and inject into workspace.
        # New sensory data starts with 0 Permanency (Ephemeral).
        sensory_chunks, cortical_state = self.senses(sensory_input)
        # print(f"sensory_chunks.shape: {sensory_chunks.shape}")
        self.memory.add_chunks(sensory_chunks, initial_permanency=self.config.INITIAL_PERMANENCY_SENSORY)

    def act(self, action_mask=None):
        """
        Phases 2-5: THINK & ACT
        Runs the cognitive cycle (Sample -> Compose -> Resolve) and returns an action.
        
        Args:
            action_mask: Optional tensor of shape (Batch, Action_Space). 
                         If provided, these values are added to the logits (e.g. -inf for illegal moves).
        """
        # --- PHASE 2: SAMPLE (Generate Context) ---
        # We need 1 Template, N Arguments, and 1 Consequent.
        # We sample randomly from the workspace.
        total_sample_size = 1 + self.config.NUM_ARGS + 1
        sample_batch, sample_indices = self.memory.get_random_subset(total_sample_size)
        
        # Unpack samples
        template = sample_batch[0].unsqueeze(0)  # (1, Dim)
        arguments = sample_batch[1:-1].unsqueeze(0) # (1, N, Dim)
        consequent = sample_batch[-1].unsqueeze(0) # (1, Dim)
        consequent_idx = sample_indices[-1]
        
        # --- PHASE 3: COMPOSE (Generate Hypothesis) ---
        # Run the binder to create M_new and its Predicted Evidence.
        m_new, predicted_cortex_new = self.composer(template, arguments)
        
        # --- PHASE 4: RESOLVE (Consistency Check) ---
        # We compare the 'Predicted Cortex' of the New Concept against the 'Predicted Cortex' of the Consequent.
        
        # 1. Generate Prediction for the Consequent
        # (In the future, we might cache this, but for now we compute on the fly)
        # We treat the consequent as a "Concept" and ask what it predicts.
        # Note: We use the same predictor. If the consequent is sensory, it should predict itself (autoencoder style).
        # But wait, our predictor takes a vector and outputs cortical state.
        # If 'consequent' is a sensory chunk, does it predict the cortical state it came from?
        # Ideally yes. The network should learn this identity mapping.
        
        # We need to project the consequent through the predictor.
        # The predictor is part of the composer, but it's just a module.
        predicted_cortex_consequent = self.composer.predictor(consequent)
        
        # 2. Check Consistency
        # Calculate Euclidean distance between the two predictions
        distance = torch.norm(predicted_cortex_new - predicted_cortex_consequent)
        
        status = "Thinking..."
        
        # 3. Update Confidence
        if distance < self.config.CONSISTENCY_THRESHOLD:
            # CONSISTENT: Both concepts agree on the state of the world.
            # Boost confidence of the Consequent.
            # (We can't boost m_new yet because it's not in memory, but we will reinforce it).
            current_conf = self.memory.confidence[consequent_idx]
            self.memory.confidence[consequent_idx] = min(1.0, current_conf + 0.1)
            
            # Reinforce the New Concept (Induction)
            self.memory.reinforce(m_new)
            status = f"Consistency Found! Dist: {distance.item():.4f}. Reinforced new concept."
            
        else:
            # INCONSISTENT: Contradiction.
            # One of them is wrong. We punish the weaker one.
            # Note: m_new doesn't have a stored confidence yet. We can assign it a default or derive from parents.
            # Let's assume m_new inherits confidence from its parents (Template/Args).
            # For now, let's just use INITIAL_CONFIDENCE_CONCEPT.
            
            conf_new = self.config.INITIAL_CONFIDENCE_CONCEPT
            conf_consequent = self.memory.confidence[consequent_idx]
            
            if conf_new > conf_consequent:
                # The New Idea is stronger than the Old Memory.
                # Demote the Old Memory.
                self.memory.confidence[consequent_idx] = max(0.0, conf_consequent - 0.1)
                # If confidence drops to zero, we might kill it (permanency = 0)
                if self.memory.confidence[consequent_idx] <= 0:
                    self.memory.permanency[consequent_idx] = 0.0
                status = f"Contradiction! Dist: {distance.item():.4f}. Demoted consequent."
            else:
                # The Old Memory is stronger. The New Idea is bad.
                # We simply discard m_new (don't reinforce it).
                status = f"Contradiction! Dist: {distance.item():.4f}. Discarded hypothesis."

        # --- PHASE 5: ACT ---
        # Action is a function of the most active/unexplained concepts.
        top_memories = self.memory.get_top_k(self.config.TOP_K_FOR_ACTION)
        action_logits = self.motor(top_memories)
        
        # Apply mask if provided
        if action_mask is not None:
            action_logits = action_logits + action_mask
            
        action = torch.distributions.Categorical(logits=action_logits).sample()
        
        return action.item(), status

if __name__ == "__main__":
    agent = RadicalAgent()
    print("Agent Initialized. Model is ready for experiments.")