class Config:
    """
    Hyperparameters for the agent.
    """
    # --- Memory Workspace ---
    MEMORY_DIM = 64         # Size of the unified vector space for all chunks (Sensory, Concepts, Goals)
    MAX_MEMORIES = 1000     # Capacity of the workspace (circular buffer)
    INITIAL_PERMANENCY_SENSORY = 0.0
    INITIAL_PERMANENCY_CONCEPT = 1.0
    
    # --- Sensory Cortex ---
    SENSORY_CHUNKS = 128      # Number of chunks to extract from senses
    INPUT_CHANNELS = 12     # 12 Channels (6 pieces * 2 colors)
    CONV1_CHANNELS = 16
    CONV2_CHANNELS = 32
    CONV_KERNEL_SIZE = 3
    CONV_STRIDE = 1         # Stride 1 to preserve resolution
    CONV_PADDING = 1        # Padding 1 to preserve resolution
    # Derived from 8x8 input with stride 1 and padding 1: 32 channels * 8 * 8 spatial
    # Derived from 8x8 input with stride 1 and padding 1: 32 channels * 8 * 8 spatial
    FLATTENED_FEATURE_SIZE = 32 * 8 * 8 
    
    # Total Cortical Dimension (Sum of all layer outputs)
    # Conv1: 16 * 8 * 8 = 1024
    # Conv2: 32 * 8 * 8 = 2048
    # Flattened: 2048
    # Total = 1024 + 2048 + 2048 = 5120
    # Note: This is a simplification. In a real ResNet, this would be huge.
    # We'll just use the flattened feature size for now as the "Cortex" to keep it simple,
    # or we can concatenate everything. Let's concatenate Conv1 + Conv2 outputs.
    CORTICAL_DIM = (16 * 8 * 8) + (32 * 8 * 8) 

    # --- Sampling Configuration ---
    # We sample a set of chunks. One acts as the 'Template' (Structure),
    # and the others act as 'Arguments' (Fillers).
    NUM_ARGS = 3            # Number of argument chunks to pull (Fillers)
    INTERNAL_SLOTS = 3      # Number of roles the Template attempts to fill (e.g., Agent, Patient, Loc)
    
    # --- Concept Composer ---
    ATTENTION_HEADS = 4
    CONCEPT_HIDDEN_DIM = 128
    
    # --- Interaction & Action ---
    INITIAL_CONFIDENCE_SENSORY = 1.0
    INITIAL_CONFIDENCE_CONCEPT = 0.5
    CONFIDENCE_DECAY = 0.01
    CONSISTENCY_THRESHOLD = 0.5 # Distance threshold for contradiction
    
    TOP_K_FOR_ACTION = 5        # Number of active memories used to drive policy
    ACTION_SPACE = 4096         # 64 * 64 possible moves (from_square * to_square)
    MOTOR_HIDDEN_DIM = 64
