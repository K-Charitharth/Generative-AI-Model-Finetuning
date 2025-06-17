# main.py
import torch
import time
from config import Config
from data_loader import DataHandler
from model_manager import ModelManager
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from synth_generator import SyntheticDataGenerator
from visualizer import ResultVisualizer

def main():
    # Initialize components
    config = Config()
    data_handler = DataHandler()
    model_manager = ModelManager()
    visualizer = ResultVisualizer()
    
    # Create directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 1. Load and prepare data
    print("Loading and splitting dataset...")
    train_data, test_data = data_handler.load_and_split_data()
    data_handler.save_dataset(train_data, "train.json")
    data_handler.save_dataset(test_data, "test.json")
    
    # 2. Load base model (Model A)
    print("Loading base model...")
    model, tokenizer = model_manager.load_base_model()
    model_a = model_manager.apply_lora(model)
    model_manager.save_tokenizer(tokenizer)
    
    # 3. Evaluate Model A
    print("Evaluating Model A...")
    evaluator = ModelEvaluator(model_a, tokenizer)
    evaluator.load_comet_model()
    tokenized_test = data_handler.preprocess_dataset(test_data, tokenizer)
    comet_results, _, _ = evaluator.compute_comet(test_data)
    comet_score_a = evaluator.average_comet_score(comet_results)
    print(f"Model A COMET Score: {comet_score_a:.4f}")
    
    # 4. Fine-tune Model A → Model B (Original data)
    print("Training Model B on original data...")
    tokenized_train = data_handler.preprocess_dataset(train_data, tokenizer)
    trainer = ModelTrainer(model_a, tokenizer)
    sft_trainer = trainer.create_trainer(tokenized_train, tokenized_test)
    model_b = trainer.train(sft_trainer)
    model_manager.save_model(model_b, "model_b")
    
    # Evaluate Model B
    evaluator.model = model_b
    comet_results, _, _ = evaluator.compute_comet(test_data)
    comet_score_b = evaluator.average_comet_score(comet_results)
    print(f"Model B COMET Score: {comet_score_b:.4f}")
    
    # 5. Generate synthetic data
    print("Generating synthetic data...")
    synth_generator = SyntheticDataGenerator(api_key="YOUR_API_KEY")  # Replace with your key
    synthetic_data = synth_generator.generate_dataset(train_data)
    synthetic_dataset = Dataset.from_list(synthetic_data)
    data_handler.save_dataset(synthetic_dataset, "synthetic.json")
    tokenized_synth = data_handler.preprocess_dataset(synthetic_dataset, tokenizer)
    
    # 6. Fine-tune Model A → Model C (Synthetic data)
    print("Training Model C on synthetic data...")
    trainer_c = ModelTrainer(model_a, tokenizer)
    sft_trainer_c = trainer_c.create_trainer(tokenized_synth, tokenized_test)
    model_c = trainer_c.train(sft_trainer_c)
    model_manager.save_model(model_c, "model_c")
    
    # Evaluate Model C
    evaluator.model = model_c
    comet_results, _, _ = evaluator.compute_comet(test_data)
    comet_score_c = evaluator.average_comet_score(comet_results)
    print(f"Model C COMET Score: {comet_score_c:.4f}")
    
    # 7. Create combined dataset
    print("Creating combined dataset...")
    combined_data = data_handler.combine_datasets(tokenized_train, tokenized_synth)
    data_handler.save_dataset(combined_data, "combined.json")
    
    # 8. Fine-tune Model A → Model D (Combined data)
    print("Training Model D on combined data...")
    trainer_d = ModelTrainer(model_a, tokenizer)
    sft_trainer_d = trainer_d.create_trainer(combined_data, tokenized_test)
    model_d = trainer_d.train(sft_trainer_d)
    model_manager.save_model(model_d, "model_d")
    
    # Evaluate Model D
    evaluator.model = model_d
    comet_results, _, _ = evaluator.compute_comet(test_data)
    comet_score_d = evaluator.average_comet_score(comet_results)
    print(f"Model D COMET Score: {comet_score_d:.4f}")
    
    # 9. Visualize results
    scores = {
        "Model A (Base)": comet_score_a,
        "Model B (Original)": comet_score_b,
        "Model C (Synthetic)": comet_score_c,
        "Model D (Combined)": comet_score_d
    }
    visualizer.plot_comet_scores(scores, "comet_scores_comparison.png")
    
    # 10. Create dataset archive
    data_handler.create_zip_archive()
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()
