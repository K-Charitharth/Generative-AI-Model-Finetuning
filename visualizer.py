# visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from .config import Config

class ResultVisualizer:
    def __init__(self):
        self.config = Config()
        
    def plot_comet_scores(self, scores_dict, save_path=None):
        """
        Plot COMET scores in a bar chart
        
        Args:
            scores_dict: Dictionary of model names to scores
            save_path: Path to save the plot (optional)
        """
        models = list(scores_dict.keys())
        scores = list(scores_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        plt.xlabel("Model Versions", fontsize=12)
        plt.ylabel("COMET Score", fontsize=12)
        plt.title("Translation Quality Comparison", fontsize=14)
        plt.ylim(min(scores)*0.95, max(scores)*1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., 
                height, 
                f"{score:.4f}", 
                ha='center', 
                va='bottom', 
                fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        return plt
