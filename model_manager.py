# model_manager.py
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from .config import Config

class ModelManager:
    def __init__(self):
        self.config = Config()
        
    def load_base_model(self):
        """Load base model with 4-bit quantization"""
        return FastLanguageModel.from_pretrained(
            model_name=self.config.MODEL_NAME,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=self.config.LOAD_IN_4BIT,
        )
    
    def apply_lora(self, model):
        """Apply LoRA to the model"""
        return FastLanguageModel.get_peft_model(
            model,
            r=self.config.LORA_R,
            target_modules=self.config.TARGET_MODULES,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    
    def save_model(self, model, model_name):
        """Save model to directory"""
        path = os.path.join(self.config.MODEL_DIR, model_name)
        model.save_pretrained(path)
        return path
    
    def load_model(self, model_name):
        """Load model from directory"""
        path = os.path.join(self.config.MODEL_DIR, model_name)
        return FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=self.config.LOAD_IN_4BIT,
        )
    
    def save_tokenizer(self, tokenizer):
        """Save tokenizer to directory"""
        path = os.path.join(self.config.MODEL_DIR, "tokenizer")
        tokenizer.save_pretrained(path)
        return path
    
    def load_tokenizer(self):
        """Load tokenizer from directory"""
        path = os.path.join(self.config.MODEL_DIR, "tokenizer")
        return AutoTokenizer.from_pretrained(path)
