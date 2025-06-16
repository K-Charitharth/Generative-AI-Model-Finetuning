# config.py
class Config:
    # Model configuration
    MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    LORA_R = 16
    LORA_ALPHA = 16
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Data configuration
    DATASET_NAME = "opus100"
    LANG_PAIR = "de-fr"
    SAMPLE_SIZE = 1000
    TRAIN_TEST_SPLIT = 0.2
    SEED = 42
    SYNTHETIC_SCALE_FACTOR = 2
    
    # Training configuration
    LEARNING_RATE = 3e-5
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    
    # Evaluation configuration
    COMET_MODEL = "Unbabel/wmt22-comet-da"
    
    # Synthetic data configuration
    TOPICS = [
        "Casual conversation", "Technology", "Healthcare", "Legal", "Business",
        "Education", "Travel", "Entertainment", "Science", "Environment", "History", "Sports"
    ]
    SYNTHETIC_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    # Path configuration
    DATA_DIR = "datasets"
    MODEL_DIR = "saved_models"
    RESULTS_DIR = "results"
