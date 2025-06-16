# data_loader.py
import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets
from .config import Config

class DataHandler:
    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        
    def load_and_split_data(self):
        """Load dataset from Hugging Face and split into train/test"""
        dataset = load_dataset(
            self.config.DATASET_NAME, 
            self.config.LANG_PAIR
        )["test"].shuffle(seed=self.config.SEED).select(range(self.config.SAMPLE_SIZE))
        
        split = dataset.train_test_split(
            test_size=self.config.TRAIN_TEST_SPLIT, 
            seed=self.config.SEED
        )
        return split["train"], split["test"]
    
    def save_dataset(self, dataset, file_name):
        """Save dataset to JSON file"""
        path = os.path.join(self.config.DATA_DIR, file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=4, ensure_ascii=False)
        return path
    
    def load_json_dataset(self, file_name):
        """Load dataset from JSON file"""
        path = os.path.join(self.config.DATA_DIR, file_name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_dict(data)
    
    def preprocess_dataset(self, dataset, tokenizer):
        """Tokenize and format dataset for training"""
        def preprocess_function(examples):
            prompts = [self.format_prompt(text['de']) for text in examples["translation"]]
            targets = [text["fr"] for text in examples["translation"]]
            model_inputs = tokenizer(prompts, max_length=256, truncation=True, padding="max_length")
            labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
            
            model_inputs["labels"] = labels["input_ids"]
            model_inputs["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in label]
                for label in model_inputs["labels"]
            ]
            return model_inputs
        
        return dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=dataset.column_names
        )
    
    @staticmethod
    def format_prompt(text):
        """Format prompt for translation"""
        return f"""<|im_start|>system
You are a professional German-to-French translator. Follow these rules STRICTLY:
1. Translate ALL text to French EXCEPT:
   - Proper nouns (names, laws, organizations)
   - Technical terms without direct equivalents
   - Numbers/measurements
2. Never add explanations or notes
3. Output ONLY the French translation
4. Maintain original formatting and punctuation
5. Never include any non-French text

Now translate:
{text}<|im_end|>
<|im_start|>assistant
"""
    
    def combine_datasets(self, dataset1, dataset2):
        """Combine two datasets"""
        return concatenate_datasets([dataset1, dataset2])
    
    def create_zip_archive(self):
        """Create ZIP archive of dataset splits"""
        files = ["train.json", "test.json", "synthetic.json", "combined.json"]
        with zipfile.ZipFile("dataset_splits.zip", "w") as zipf:
            for file in files:
                path = os.path.join(self.config.DATA_DIR, file)
                if os.path.exists(path):
                    zipf.write(path)
        return "dataset_splits.zip"
