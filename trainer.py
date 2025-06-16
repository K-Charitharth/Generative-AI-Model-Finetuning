# trainer.py
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from .config import Config

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = Config()
        
    def create_trainer(self, train_dataset, eval_dataset):
        """Create SFTTrainer with configured parameters"""
        use_bf16 = is_bfloat16_supported()
        
        training_args = TrainingArguments(
            output_dir=self.config.RESULTS_DIR,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            learning_rate=self.config.LEARNING_RATE,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRAD_ACCUM_STEPS,
            num_train_epochs=self.config.NUM_EPOCHS,
            weight_decay=self.config.WEIGHT_DECAY,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=10,
            report_to="none",
            optim="adamw_8bit",
            warmup_ratio=self.config.WARMUP_RATIO,
            max_grad_norm=self.config.MAX_GRAD_NORM
        )
        
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model, 
            label_pad_token_id=self.tokenizer.pad_token_id
        )
        
        return SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            packing=True
        )
    
    def train(self, trainer):
        """Execute training process"""
        trainer.train()
        return trainer.model
