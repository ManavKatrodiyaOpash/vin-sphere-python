import os

# Paths
DATA_PATH = "Data/Cleaned.csv"
MODEL_DIR = "models"
REPORTS_DIR = "outputs/reports"
ATTENTION_DIR = "outputs/attention_maps"

# Tokenizer Configuration
ALLOWED_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

# Model Hyperparameters
EMBED_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
ATTENTION_POOLING_DIM = 128

# Target column classifications and regression configurations
CLASSIFICATION_TARGETS = [
    "year", "regionalSpec", "origin", "make_grouped",
    "model_final", "bodyType_raw", "trim_raw", "color_raw"
]

REGRESSION_TARGETS = [
    "weightInKg", "noOfPassengers"
]

ALL_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS

# Hierarchical Stages for prediction
HIERARCHICAL_STAGES = [
    "make_grouped",  # Stage 1
    "model_final",   # Stage 2
    "trim_raw",      # Stage 3
    "bodyType_raw"   # Stage 4
]

# Training configuration
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 1e-2

# Imbalance threshold (rare class grouping)
RARE_THRESHOLD = 5

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ATTENTION_DIR, exist_ok=True)
