# synth_generator.py
import json
import requests
import time
import random
from .config import Config

class SyntheticDataGenerator:
    def __init__(self, api_key):
        self.config = Config()
        self.API_URL = "https://api.together.xyz/v1/chat/completions"
        self.HEADERS = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json"
        }
        self.PROMPT_TEMPLATE = """
Generate a synthetic German-French parallel sentence pair based on the following category: {topic}.
Ensure the sentence is natural, well-formed, and contextually relevant.

Example input sentence:
German: "{german_sentence}"
French: "{french_sentence}"

Now generate a NEW synthetic German-French sentence pair on the topic '{topic}'.
Provide only the output in JSON format:
{{
    "german": "...",
    "french": "..."
}}
"""
    
    def generate_synthetic_translation(self, german, french):
        """Generate a single synthetic translation pair"""
        topic = random.choice(self.config.TOPICS)
        prompt = self.PROMPT_TEMPLATE.format(
            german_sentence=german, 
            french_sentence=french, 
            topic=topic
        )
        
        payload = {
            "model": self.config.SYNTHETIC_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.API_URL, headers=self.HEADERS, json=payload)
            response.raise_for_status()
            response_json = response.json()
            generated_text = response_json["choices"][0]["message"]["content"].strip()
            synthetic_pair = json.loads(generated_text)
            return synthetic_pair.get("german", ""), synthetic_pair.get("french", "")
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return "", ""
    
    def generate_dataset(self, base_dataset, scale_factor=None):
        """Generate synthetic dataset from base dataset"""
        if scale_factor is None:
            scale_factor = self.config.SYNTHETIC_SCALE_FACTOR
        
        synthetic_data = []
        for i, sample in enumerate(base_dataset):
            german_text = sample["translation"]["de"]
            french_text = sample["translation"]["fr"]
            
            for _ in range(scale_factor):
                synthetic_german, synthetic_french = self.generate_synthetic_translation(
                    german_text, french_text
                )
                if synthetic_german and synthetic_french:
                    synthetic_data.append({
                        "translation": {
                            "de": synthetic_german, 
                            "fr": synthetic_french
                        }
                    })
            
            # Rate limiting
            time.sleep(1)
            
            # Early stopping if we have enough data
            if len(synthetic_data) >= scale_factor * len(base_dataset):
                break
                
        return synthetic_data
