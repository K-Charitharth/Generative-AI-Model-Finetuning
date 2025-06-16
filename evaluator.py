# evaluator.py
from comet import download_model, load_from_checkpoint
from .config import Config
from .data_loader import DataHandler

class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = Config()
        self.data_handler = DataHandler()
        self.comet_model = None
        
    def load_comet_model(self):
        """Download and load COMET evaluation model"""
        comet_model_path = download_model(self.config.COMET_MODEL)
        self.comet_model = load_from_checkpoint(comet_model_path)
        return self.comet_model
    
    def compute_comet(self, dataset):
        """Compute COMET scores for model translations"""
        translations = []
        sources = []
        references = []
        
        for ex in dataset:
            prompt = self.data_handler.format_prompt(ex["translation"]["de"])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            translation = full_output.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
            
            translations.append(translation)
            sources.append(ex["translation"]["de"])
            references.append(ex["translation"]["fr"])
        
        comet_inputs = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(sources, translations, references)]
        comet_scores = self.comet_model.predict(comet_inputs, batch_size=8)
        return comet_scores, translations, references
    
    def average_comet_score(self, comet_scores):
        """Calculate average COMET score"""
        return sum(comet_scores['scores']) / len(comet_scores['scores'])
