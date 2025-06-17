# app.py
import gradio as gr
from .model_manager import ModelManager
from .data_loader import DataHandler

class TranslationInterface:
    def __init__(self, model_path="model_d"):
        self.model_manager = ModelManager()
        self.data_handler = DataHandler()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.model_manager.load_model(model_path)
        self.tokenizer = self.model_manager.load_tokenizer()
        
        # Move model to device
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def translate(self, text):
        """Translate German text to French"""
        if not text.strip():
            return "Please enter German text to translate"
        
        prompt = self.data_handler.format_prompt(text)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        full_output = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=False
        )
        
        # Extract translation from model output
        if "<|im_start|>assistant" in full_output:
            translation = full_output.split("<|im_start|>assistant")[1]
            if "<|im_end|>" in translation:
                return translation.split("<|im_end|>")[0].strip()
        return full_output
    
    def launch(self, share=False):
        """Launch Gradio interface"""
        interface = gr.Interface(
            fn=self.translate,
            inputs=gr.Textbox(
                label="German Input", 
                lines=5, 
                placeholder="Enter German text here..."
            ),
            outputs=gr.Textbox(
                label="French Translation", 
                lines=5
            ),
            title="🇩🇪→🇫🇷 Professional Translator",
            description="German to French translation using fine-tuned Qwen2.5-1.5B-Instruct",
            examples=[
                ["Die Umsetzung der Richtlinie 95/46/EG wird überprüft."],
                ["Der technische Fortschritt erfordert neue regulatorische Rahmenbedingungen."],
                ["Die Patientin zeigte signifikante Verbesserungen nach der Therapie."]
            ],
            css=".gradio-container {background-color: #f0f2f6;}",
            allow_flagging="never"
        )
        return interface.launch(share=share)

# For standalone execution
if __name__ == "__main__":
    import torch
    translator = TranslationInterface()
    translator.launch(share=True)
